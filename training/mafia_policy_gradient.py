"""Token- and decision-normalized policy gradient, independent of game/UI imports."""
import torch
import torch.nn.functional as F


def sequence_log_prob(model, tokenizer, device, prompt, response):
    # Match generation's left-truncated context; tokenize the action separately
    # so a tokenizer merge at the prompt boundary cannot mask action tokens.
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][:, -1536:]
    response_ids = tokenizer(response, add_special_tokens=False,
                                  return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    target_ids = input_ids.clone()
    target_ids[:, :prompt_ids.shape[1]] = -100

    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    return -loss.sum() / (shift_labels != -100).sum().clamp_min(1)


def update_policy(model, optimizer, device, learner_player, group_results, compute_log_probs):
    group_results = [r for r in group_results if not r.truncated and
                     any(t.trainable for t in r.trajectories[learner_player].turns)]
    if len(group_results) < 2:
        return 0.0

    rewards = torch.tensor(
        [res.trajectories[learner_player].shaped_reward for res in group_results],
        device=device,
        dtype=torch.float32,
    )
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r

    optimizer.zero_grad()
    total_loss = 0.0

    for i, res in enumerate(group_results):
        adv = advantages[i]
        traj = res.trajectories[learner_player]

        decisions = [t for t in traj.turns if t.trainable]
        for turn in decisions:
            log_prob = compute_log_probs(turn.prompt, turn.response)
            # Policy gradient loss: - advantage * log_prob
            loss = -adv * log_prob / (len(decisions) * len(group_results))
            loss.backward()
            total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss
