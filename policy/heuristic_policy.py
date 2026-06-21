"""Minimal team-aware heuristic policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from game.cards import Carta
from game.observation import Osservazione
from game.rules import vincitore_presa

from .greedy_policy import GreedyPolicy


@dataclass
class HeuristicPolicy:
    """Greedy policy that avoids spending cards when the partner is taking."""

    name: str = "heuristic"
    greedy_policy: GreedyPolicy = field(default_factory=GreedyPolicy)

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        carte_migliori = self._carte_migliori(osservazione)
        probabilita = 1.0 / len(carte_migliori)
        return {
            carta: probabilita if carta in carte_migliori else 0.0
            for carta in osservazione.azioni_legali
        }

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        return rng.choice(self._carte_migliori(osservazione))

    def _carte_migliori(self, osservazione: Osservazione) -> list[Carta]:
        if self._compagno_sta_prendendo(osservazione):
            return self._scarti_meno_costosi(osservazione)
        return self.greedy_policy._carte_migliori(osservazione)

    def _compagno_sta_prendendo(self, osservazione: Osservazione) -> bool:
        if not osservazione.carte_sul_campo:
            return False
        vincitore = vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id == osservazione.compagno_id

    def _scarti_meno_costosi(self, osservazione: Osservazione) -> list[Carta]:
        azioni_legali = osservazione.azioni_legali
        if not azioni_legali:
            raise ValueError("No legal actions available")
        costo_minimo = min(
            self.greedy_policy._costo_carta(osservazione, carta)
            for carta in azioni_legali
        )
        return [
            carta
            for carta in azioni_legali
            if self.greedy_policy._costo_carta(osservazione, carta) == costo_minimo
        ]
