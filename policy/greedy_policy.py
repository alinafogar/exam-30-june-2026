"""Greedy baseline policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from game.rules import vincitore_presa


@dataclass
class GreedyPolicy:
    """Myopic policy that wins with the cheapest card, otherwise discards."""

    name: str = "greedy"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        best_cards = self._best_cards(osservazione)
        probability = 1.0 / len(best_cards)
        return {
            carta: probability if carta in best_cards else 0.0
            for carta in osservazione.azioni_legali
        }

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        return rng.choice(self._best_cards(osservazione))

    def _best_cards(self, osservazione: Osservazione) -> list[Carta]:
        azioni_legali = osservazione.azioni_legali
        if not azioni_legali:
            raise ValueError("No legal actions available")

        winning_cards = [
            carta for carta in azioni_legali if self._candidate_wins(osservazione, carta)
        ]
        candidates = winning_cards or list(azioni_legali)
        best_cost = min(self._card_cost(osservazione, carta) for carta in candidates)
        return [
            carta
            for carta in candidates
            if self._card_cost(osservazione, carta) == best_cost
        ]

    def _candidate_wins(self, osservazione: Osservazione, carta: Carta) -> bool:
        candidate_trick = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=osservazione.giocatore_id, carta=carta),
        )
        winner = vincitore_presa(
            candidate_trick,
            seme_briscola=osservazione.seme_briscola,
        )
        return winner.giocatore_id == osservazione.giocatore_id

    def _card_cost(self, osservazione: Osservazione, carta: Carta) -> tuple[int, bool, int]:
        return (
            carta.punti,
            carta.seme == osservazione.seme_briscola,
            carta.forza,
        )
