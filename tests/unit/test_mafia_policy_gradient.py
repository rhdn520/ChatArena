import unittest
from types import SimpleNamespace


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is needed for tiny-model gradient checks")
class TestPolicyGradient(unittest.TestCase):
    def trainer(self):
        from training.mafia_policy_gradient import sequence_log_prob, update_policy
        trainer = SimpleNamespace()
        trainer.args = SimpleNamespace(dry_run=False)
        trainer.device = "cpu"
        trainer.learner_player = "A"
        class Tokenizer:
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = ([1] if add_special_tokens else []) + [2 + ord(c) % 13 for c in text]
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.zeros(16, 16))
            def forward(self, ids):
                return SimpleNamespace(logits=self.table[ids])
        trainer.tokenizer = Tokenizer()
        trainer.model = TinyModel()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
        trainer.compute_sequence_log_probs = lambda p, r: sequence_log_prob(
            trainer.model, trainer.tokenizer, trainer.device, p, r)
        trainer.train_step_grpo = lambda results: update_policy(
            trainer.model, trainer.optimizer, trainer.device, trainer.learner_player,
            results, trainer.compute_sequence_log_probs)
        return trainer

    def test_token_average_and_prompt_mask(self):
        trainer = self.trainer()
        short = trainer.compute_sequence_log_probs("prompt", "0")
        long = trainer.compute_sequence_log_probs("prompt", "0000")
        self.assertAlmostEqual(short.item(), long.item(), places=6)
        short.backward()
        grad = trainer.model.table.grad
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual((grad.abs().sum(dim=1) > 0).sum().item(), 1)

    def test_training_filters_records_and_moves_weights(self):
        trainer = self.trainer()
        def result(reward, response, truncated=False):
            tr = SimpleNamespace(raw_reward=reward, shaped_reward=reward)
            tr.turns = [SimpleNamespace(prompt="prompt", response=response, trainable=True),
                        SimpleNamespace(prompt="prompt", response="bad", trainable=False)]
            return SimpleNamespace(trajectories={"A": tr}, truncated=truncated)
        group = [result(1, "3"), result(-1, "0"), result(100, "2", truncated=True)]
        calls = []
        original = trainer.compute_sequence_log_probs
        def track(prompt, response):
            calls.append(response)
            return original(prompt, response)
        trainer.compute_sequence_log_probs = track
        before = trainer.model.table.detach().clone()
        loss = trainer.train_step_grpo(group)
        self.assertEqual(calls, ["3", "0"])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, trainer.model.table.detach()))
        self.assertEqual(trainer.train_step_grpo([group[0]]), 0.0)
