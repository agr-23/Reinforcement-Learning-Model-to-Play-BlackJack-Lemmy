"""
Blackjack Environment (US rules): H17, peek, no surrender.
Actions: 0=Hit (H), 1=Stand (S), 2=Double (D), 3=Split (P)

Key rules:
- H17: Dealer hits on soft 17 (e.g., A+6).
- Peek: If dealer upcard is 10/A, dealer peeks for blackjack before player acts.
- No surrender (R) not supported.
- Splits: allowed up to max_hands (default 4). 10,J,Q,K are all valued as 10 and can be split.
- Double After Split (DAS): allowed.
- Resplit Aces: NOT allowed.
- Split Aces: each split Ace receives exactly 1 card and then stands automatically; resulting 21 is NOT blackjack.
- Blackjack pays 3:2; push returns 0; normal win/loss ±1; doubles multiply payout by 2.
- Deck shoe: configurable n_decks (default 6 to mirror your CLI). Optional burn_first_card.

State format (returned in step/reset):
    dict with fields:
        'player_total' (int),
        'usable_ace' (0/1),
        'dealer_up' (int 2..11, where 11 is Ace),
        'can_double' (0/1),
        'can_split' (0/1),
        'true_count' (int), 'run_count' (int), 'cards_remaining' (int),
        'hand_index' (int),  # index of the current sub-hand being played
        'num_hands' (int),   # total sub-hands this round

The environment manages multiple sub-hands after SPLIT. Each `step()` applies to the
current sub-hand. `done=True` only when ALL sub-hands are resolved.

Reward:
- 0 for non-terminal steps.
- terminal per sub-hand according to outcome and whether it was doubled (×2).
- aggregated over sub-hands; each terminal sub-hand will return its own reward.
"""

from dataclasses import dataclass
import random
from typing import List, Tuple, Optional, Dict

# ---- Constants ----
ACTION_HIT   = 0
ACTION_STAND = 1
ACTION_DOUBLE= 2
ACTION_SPLIT = 3

CARD_VALUES = [2,3,4,5,6,7,8,9,10,10,10,10,11]  # 11 = Ace
BJ_PAYOUT = 1.5

@dataclass
class Rules:
    n_decks: int = 6
    h17: bool = True                 # dealer hits soft 17
    peek: bool = True                # US peek
    allow_surrender: bool = False    # NO surrender in US
    allow_double_any: bool = True
    allow_das: bool = True           # double after split
    max_hands: int = 4               # max hands after splits
    allow_resplit_aces: bool = False
    split_aces_one_card: bool = True # after split A, one card only, forced stand
    burn_first_card: bool = True
    blackjack_payout: float = BJ_PAYOUT

def _hand_total(cards: List[int]) -> Tuple[int, bool]:
    """Return best total and whether it's soft (at least one Ace counted as 11)."""
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    soft = aces > 0 and (total <= 21)
    return total, soft

def _is_blackjack(cards: List[int]) -> bool:
    # Natural BJ: exactly 2 cards, one Ace and one 10-value
    return len(cards) == 2 and ((11 in cards) and (10 in cards))

def _clone_list(x: List[int]) -> List[int]:
    return list(x)

class BlackjackEnv:
    def __init__(self, rules: Rules = Rules(), seed: Optional[int] = None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.shoe: List[int] = []
        self.run_count = 0
        self.true_count = 0
        self._reset_shoe()

        # Round state
        self.dealer: List[int] = []
        self.hands: List[List[int]] = []       # list of player sub-hands
        self.hands_done: List[bool] = []
        self.hands_doubled: List[bool] = []
        self.hands_is_split_aces: List[bool] = []
        self.current = 0                       # index of current hand being played
        self.episode_done = False

    # ---------- Shoe & dealing ----------
    def _reset_shoe(self):
        self.shoe = []
        for _ in range(self.rules.n_decks * 4):  # 4 suits
            self.shoe.extend(CARD_VALUES)
        self.rng.shuffle(self.shoe)
        if self.rules.burn_first_card and self.shoe:
            self._draw_card()  # burn one

        self.run_count = 0
        self.true_count = 0  # approximate TC not modeled precisely; left as 0

    def cards_remaining(self) -> int:
        return len(self.shoe)

    def _draw_card(self) -> int:
        if not self.shoe:
            self._reset_shoe()
        card = self.shoe.pop()
        # Hi-Lo running count update (optional; keep simple)
        # 2-6 -> +1, 7-9 -> 0, T-A -> -1
        if card in (2,3,4,5,6): self.run_count += 1
        elif card in (10,11):   self.run_count -= 1
        # true_count could be estimated as run_count / decks_remaining (approx)
        decks_rem = max(0.25, len(self.shoe)/52.0)
        self.true_count = int(self.run_count / decks_rem)
        return card

    # ---------- Round flow ----------
    def reset(self) -> Dict:
        """Start a new round: deal two cards to player, two to dealer. Handle peek/BJ."""
        self.dealer = [self._draw_card(), self._draw_card()]
        p = [_clone_list([self._draw_card(), self._draw_card()])]
        self.hands = p
        self.hands_done = [False]
        self.hands_doubled = [False]
        # mark split-aces flag per hand (initially false)
        self.hands_is_split_aces = [False]
        self.current = 0
        self.episode_done = False

        # Peek: if dealer upcard is 10/A, check BJ now
        dealer_up = self.dealer[0]
        if self.rules.peek and dealer_up in (10, 11):
            if _is_blackjack(self.dealer):
                # Dealer BJ: end round immediately; resolve vs player's BJ (push) or loss
                reward = 0.0
                for i, hand in enumerate(self.hands):
                    if _is_blackjack(hand):
                        # push
                        reward += 0.0
                    else:
                        reward += -1.0
                    self.hands_done[i] = True
                self.episode_done = True
                return self._obs(), reward, True, {"reason": "dealer_blackjack_peek"}

        # Natural player BJ without dealer BJ (after peek): immediate payout for that hand
        if _is_blackjack(self.hands[0]):
            # Pay BJ unless coming from split (not possible at reset)
            reward = self.rules.blackjack_payout
            self.hands_done[0] = True
            self.episode_done = True
            return self._obs(), reward, True, {"reason": "player_blackjack"}

        return self._obs(), 0.0, False, {}

    def available_actions(self) -> List[int]:
        """Actions allowed for the CURRENT hand."""
        if self.episode_done:
            return []
        i = self.current
        hand = self.hands[i]
        total, _ = _hand_total(hand)

        acts = [ACTION_HIT, ACTION_STAND]
        # Double allowed if exactly 2 cards; DAS allowed → after split too
        if self.rules.allow_double_any and len(hand) == 2:
            acts.append(ACTION_DOUBLE)
        # Split allowed if exactly 2 cards and equal value and we still can create a new hand
        if len(hand) == 2 and self._can_split_pair(hand):
            if len(self.hands) < self.rules.max_hands:
                acts.append(ACTION_SPLIT)
        # If split Aces one card only and already got one extra, force stand (no H/D)
        if self.hands_is_split_aces[i] and len(hand) >= 2:
            # After split of A, we add exactly one card and must stand; override
            return [ACTION_STAND]
        # If total >= 21, only stand is meaningful
        if total >= 21:
            return [ACTION_STAND]
        return acts

    def _can_split_pair(self, hand: List[int]) -> bool:
        # ten-valued cards can split regardless of face
        if len(hand) != 2: return False
        v1, v2 = hand
        if v1 == 10 and v2 == 10:
            return True
        if v1 == 11 and v2 == 11:
            # resplit aces?
            if self.hands_is_split_aces[self.current] and not self.rules.allow_resplit_aces:
                return False
            return True
        return v1 == v2

    def _obs(self) -> Dict:
        i = self.current
        hand = self.hands[i]
        total, soft = _hand_total(hand)
        dealer_up = self.dealer[0]
        acts = self.available_actions()
        can_double = int(ACTION_DOUBLE in acts)
        can_split  = int(ACTION_SPLIT in acts)
        return dict(
            player_total=total,
            usable_ace=1 if soft else 0,
            dealer_up=dealer_up,
            can_double=can_double,
            can_split=can_split,
            true_count=self.true_count,
            run_count=self.run_count,
            cards_remaining=self.cards_remaining(),
            hand_index=i,
            num_hands=len(self.hands),
        )

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Apply action to the CURRENT hand. Returns (obs, reward, done, info)."""
        assert not self.episode_done, "Episode is done. Call reset()."
        assert action in (ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT), "Invalid action"

        i = self.current
        hand = self.hands[i]
        info: Dict = {}

        if action not in self.available_actions():
            # Illegal action → ignore and force STAND to be safe
            action = ACTION_STAND
            info["illegal_action_forced_to_stand"] = True

        reward = 0.0
        done_sub = False

        if action == ACTION_SPLIT:
            # Split current hand into two
            c1, c2 = hand
            new1 = [c1, self._draw_card()]
            new2 = [c2, self._draw_card()]
            # Replace current with first, append second
            self.hands[i] = new1
            self.hands.insert(i+1, new2)
            self.hands_done[i] = False
            self.hands_done.insert(i+1, False)
            self.hands_doubled[i] = False
            self.hands_doubled.insert(i+1, False)

            # Mark if split-aces
            is_aces = (c1 == 11 and c2 == 11)
            self.hands_is_split_aces[i] = is_aces
            self.hands_is_split_aces.insert(i+1, is_aces)

            # If split Aces one card: both hands auto-stand immediately
            if is_aces and self.rules.split_aces_one_card:
                self.hands_done[i] = True
                self.hands_done[i+1] = True
                # move to next unresolved hand or resolve dealer if all done
                self._advance_or_resolve()
                return self._obs(), 0.0, self.episode_done, {"split_aces_forced_stand": True}

            # Otherwise, continue playing current hand (now with a new card)
            return self._obs(), 0.0, False, {"split": True}

        if action == ACTION_DOUBLE:
            # Take 1 card then stand
            hand.append(self._draw_card())
            self.hands_doubled[i] = True
            done_sub = True

        elif action == ACTION_HIT:
            hand.append(self._draw_card())
            total, _ = _hand_total(hand)
            if total >= 21:  # 21 or bust → auto-stand (terminal for this sub-hand)
                done_sub = True

        elif action == ACTION_STAND:
            done_sub = True

        # If sub-hand finished, check if we should advance to next or resolve vs dealer
        if done_sub:
            self.hands_done[i] = True
            # If split A, ensure we respect one-card-after-split rule (should be already handled)
            self._advance_or_resolve()

            # If the episode ended after resolving dealer, reward is attached in info; compute and return here
            # But we compute rewards only when dealer finished & outcomes known.
            if "subhand_reward" in self._last_info:
                reward = self._last_info["subhand_reward"]
                info.update(self._last_info)
                self._last_info = {}
            return self._obs(), reward, self.episode_done, info

        # Otherwise, keep playing same sub-hand
        return self._obs(), 0.0, False, info

    # ---------- Resolution ----------
    _last_info: Dict = {}

    def _advance_or_resolve(self):
        """Move to next unresolved hand if any; otherwise play dealer and score all hands.
           Store the reward for the sub-hand that just ended in _last_info['subhand_reward']."""
        # Move to next unresolved
        for j in range(len(self.hands)):
            if not self.hands_done[j]:
                self.current = j
                self._last_info = {}
                return
        # All sub-hands are done; resolve dealer and compute payouts for EACH sub-hand
        self._play_dealer()
        # Score current sub-hand (the one that just finished) and return its reward;
        # however, since step() is called per sub-hand, we compute the reward for the sub-hand that just ended.
        # To keep it simple and deterministic, we compute reward for the sub-hand that just ended (self.current),
        # then move pointer to next (if any). But we are here because all are done, so self.current stays.
        # We'll compute the reward for the last finished sub-hand and store it. The environment ends here.
        last_idx = self.current
        sub_rewards = self._score_all_subhands()
        self._last_info = {"subhand_reward": sub_rewards[last_idx]}
        self.episode_done = True

    def _play_dealer(self):
        """Dealer plays according to rules after all player sub-hands finished."""
        # If dealer has blackjack from two cards, nothing else to do (already handled by peek at reset).
        # Dealer draws until 17 or more; if H17 and soft 17, must hit.
        while True:
            total, soft = _hand_total(self.dealer)
            if total < 17:
                self.dealer.append(self._draw_card())
                continue
            if total == 17 and soft and self.rules.h17:
                self.dealer.append(self._draw_card())
                continue
            break

    def _score_all_subhands(self) -> List[float]:
        """Return a list of rewards for each sub-hand."""
        rewards: List[float] = []
        dealer_total, _ = _hand_total(self.dealer)
        dealer_bust = dealer_total > 21

        for i, hand in enumerate(self.hands):
            total, _ = _hand_total(hand)
            doubled = self.hands_doubled[i]

            # Check if this hand is a "natural BJ" (2-card 21) and not from split-aces
            is_bj = _is_blackjack(hand) and not self.hands_is_split_aces[i]

            if is_bj and len(hand) == 2:
                # Natural BJ vs dealer:
                if _is_blackjack(self.dealer):
                    reward = 0.0
                else:
                    reward = self.rules.blackjack_payout
            else:
                if total > 21:
                    reward = -1.0
                elif dealer_bust:
                    reward = 1.0
                else:
                    if total > dealer_total:
                        reward = 1.0
                    elif total < dealer_total:
                        reward = -1.0
                    else:
                        reward = 0.0
            if doubled:
                reward *= 2.0
            rewards.append(reward)
        return rewards

    # ---------- Convenience ----------
    def render(self) -> str:
        def fmt(cards: List[int]) -> str:
            # Convert 11→A and 10 stays 10 (stands for 10/J/Q/K)
            def cv(x): return 'A' if x == 11 else str(x)
            return " ".join(cv(c) for c in cards)
        s = []
        s.append(f"Dealer: {fmt(self.dealer)}")
        for i, h in enumerate(self.hands):
            cur = " <=" if i == self.current and not self.episode_done else ""
            s.append(f"Hand {i+1}/{len(self.hands)}: {fmt(h)}{cur}")
        return "\n".join(s)