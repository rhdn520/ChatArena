import threading
import unittest

from chatarena.environments.mafia import Mafia
from chatarena.mafia_discussion import MafiaDiscussionController
from test_mafia_discussion import drive

NAMES = list('ABCDEFG')
ROLES = dict(zip(NAMES, ['mafia', 'mafia', 'doctor', 'police', 'villager', 'villager', 'villager']))


class TestPrivateDiscussion(unittest.TestCase):
    def controller(self, env, query, **kwargs):
        ctrl = MafiaDiscussionController(env, query, **kwargs)
        self.addCleanup(ctrl.close)
        return ctrl

    def test_partners_known_at_assignment_before_any_discussion(self):
        env = Mafia(NAMES, role_mapping=ROLES)
        for _ in range(2):
            for name, partner in [('A', 'B'), ('B', 'A')]:
                own = env.get_observation(name)
                self.assertTrue(any(f'동료 마피아: {partner}.' in m.content for m in own))
                self.assertFalse(any('동료 마피아:' in m.content for m in env.get_observation('G')))
            self.assertFalse(any(m.agent_name in ('A', 'B') for m in env.get_observation()))
            self.assertEqual(env.phase, 'NIGHT_MAFIA_DISCUSSION')
            env.reset()

    def test_private_exchange_then_vote_and_day(self):
        env = Mafia(NAMES, role_mapping=ROLES, mafia_discussion_messages=2)
        seen = []
        def query(req):
            seen.append(req)
            if req.kind == 'intent':
                return '3' if req.player_name == ('A' if env.discussion_messages == 0 else 'B') else '0'
            return 'E를 공격하자' if req.player_name == 'A' else '그 의견에 동의해'
        def forbidden(req):
            self.fail('Moderator must never run in private discussion')
        ctrl = self.controller(env, query, moderator_query=forbidden)
        drive(ctrl, lambda: env.phase == 'NIGHT_MAFIA')
        self.assertEqual(set(r.player_name for r in seen), {'A', 'B'})
        b_speech = next(r for r in seen if r.kind == 'speech' and r.player_name == 'B')
        self.assertTrue(any(m.content == 'E를 공격하자' for m in b_speech.observation))
        self.assertTrue(all(m.visible_to == 'all' or r.player_name in m.visible_to
                            for r in seen for m in r.observation))
        for citizen in 'CDEFG':
            self.assertFalse(any(m.content in ('E를 공격하자', '그 의견에 동의해')
                                 for m in env.get_observation(citizen)))
        self.assertTrue(all(r.trainable for r in ctrl.records))
        env.step('A', 'E를 제거하겠습니다')
        env.step('B', 'E를 제거하겠습니다')
        self.assertEqual(env.night_kills, ['E'])
        env.step('C', 'G를 보호하겠습니다')
        env.step('D', 'A를 조사하겠습니다')
        self.assertEqual(env.phase, 'DAY_DISCUSSION')
        self.assertNotIn('E', env.alive_players)
        self.assertEqual(env.discussion_messages, 0)
        self.assertEqual(env.discussion_endings[0]['phase'], 'NIGHT_MAFIA_DISCUSSION')

    def test_pass_and_zero_end_private_discussion(self):
        for score in ('0', '2'):
            env = Mafia(NAMES, role_mapping=ROLES)
            ctrl = self.controller(env, lambda req: score if req.kind == 'intent' else 'PASS')
            drive(ctrl, lambda: env.phase == 'NIGHT_MAFIA')
            self.assertFalse(any(m.content == 'PASS' for m in env.get_observation()))
            self.assertEqual(env.discussion_endings[-1]['reason'], 'silence')

    def test_majority_and_seeded_ties(self):
        names = list('ABCDEFGHI')
        roles = {n: 'mafia' if n in 'ABC' else 'villager' for n in names}
        env = Mafia(names, role_mapping=roles)
        env.end_discussion('test')
        for actor, target in [('A', 'D'), ('B', 'D'), ('C', 'E')]:
            env.step(actor, target)
        self.assertNotIn('D', env.alive_players)
        self.assertIn('E', env.alive_players)
        outcomes = []
        for _ in range(2):
            env = Mafia(NAMES, role_mapping=ROLES, seed=42)
            env.end_discussion('test')
            env.step('A', 'E')
            env.step('B', 'F')
            outcomes.append(env.night_kills[0])
            self.assertIn(outcomes[-1], ['E', 'F'])
            self.assertFalse(any('최종 공격 대상' in m.content
                                 for m in env.get_observation('G')))
        self.assertEqual(*outcomes)

    def test_citizen_rejected_and_human_interrupt_invalidates_speech(self):
        env = Mafia(NAMES, role_mapping=ROLES)
        entered, release = threading.Event(), threading.Event()
        def query(req):
            if req.kind == 'intent':
                return '3'
            entered.set()
            release.wait(2)
            return '오래된 제안'
        ctrl = self.controller(env, query, human_names=['A', 'G'])
        self.addCleanup(release.set)
        with self.assertRaises(ValueError):
            ctrl.submit_human('G', '비공개방 침입')
        drive(ctrl, entered.is_set)
        ctrl.submit_human('A', '대상을 다시 생각하자')
        release.set()
        drive(ctrl, lambda: any(r.stale for r in ctrl.records))
        self.assertFalse(any(m.content == '오래된 제안' for m in env.get_observation()))
        self.assertFalse(any(m.content == '대상을 다시 생각하자' for m in env.get_observation('G')))
        self.assertTrue(all(not r.trainable for r in ctrl.records if r.stale))

    def test_single_surviving_mafia_skips_discussion(self):
        env = Mafia(NAMES, role_mapping=ROLES)
        env.alive_players.remove('B')
        env._start_night_phase()
        self.assertEqual(env.phase, 'NIGHT_MAFIA')
        self.assertEqual(env.get_next_player(), 'A')

    def test_private_decisions_in_completed_rl_trajectory(self):
        from chatarena.rl.mafia_rollout import MafiaRolloutManager
        def policy(name, prompt):
            if 'REQUEST intent:' in prompt:
                return '2' if '현재는 밤의 마피아 전용' in prompt else '0'
            if 'REQUEST speech:' in prompt:
                return '비밀 작전: G를 공격하자'
            if 'NIGHT_MAFIA입니다' in prompt:
                return 'G를 제거하겠습니다'
            if 'NIGHT_DOCTOR입니다' in prompt:
                return 'G를 보호하겠습니다'
            return 'A'
        manager = MafiaRolloutManager(player_names=NAMES, role_mapping=ROLES,
                                      mafia_discussion_messages=1, max_days=1)
        result = manager.rollout_episode(NAMES, policy, policy)
        self.assertFalse(result.truncated)
        for name in ('A', 'B'):
            private = [t for t in result.trajectories[name].turns
                       if t.phase == 'NIGHT_MAFIA_DISCUSSION']
            self.assertTrue(private)
            self.assertTrue(all(t.trainable for t in private))
        for name in 'CDEFG':
            self.assertFalse(any('비밀 작전' in t.prompt for t in result.trajectories[name].turns))

    def test_human_silence_timeout_and_private_time_limit(self):
        for cap, reason in [(60, 'silence'), (1, 'time_limit')]:
            now = [0.0]
            env = Mafia(NAMES, role_mapping=ROLES, mafia_discussion_seconds=cap)
            ctrl = self.controller(env, lambda req: '0', human_names=['A'],
                                   realtime=True, clock=lambda: now[0])
            drive(ctrl, lambda: ctrl.status == 'waiting_human')
            self.assertEqual(env.phase, 'NIGHT_MAFIA_DISCUSSION')
            now[0] = 11
            ctrl.tick()
            self.assertEqual(env.phase, 'NIGHT_MAFIA')
            self.assertEqual(env.discussion_endings[-1]['reason'], reason)
