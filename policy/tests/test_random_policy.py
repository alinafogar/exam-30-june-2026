from __future__ import annotations

import random
import unittest

from game.cards import Carta
from game.observation import Osservazione
from policy import RandomPolicy


def osservazione(mano: tuple[Carta, ...] | None = None) -> Osservazione:
    if mano is None:
        mano = (
            Carta("coppe", "asso"),
            Carta("bastoni", "due"),
            Carta("denari", "tre"),
        )

    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola="denari",
        briscola_esposta=Carta("denari", "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=(),
        carte_giocate=(),
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=0,
    )


class TestRandomPolicy(unittest.TestCase):
    def test_action_probabilities_are_uniform(self):
        obs = osservazione()
        policy = RandomPolicy()

        probabilities = policy.action_probabilities(obs)

        self.assertEqual(set(probabilities), set(obs.azioni_legali))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        for probability in probabilities.values():
            self.assertAlmostEqual(probability, 1.0 / 3)

    def test_select_action_returns_legal_card(self):
        obs = osservazione()
        policy = RandomPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertIn(action, obs.azioni_legali)

    def test_empty_legal_actions_raise_value_error(self):
        obs = osservazione(mano=())
        policy = RandomPolicy()

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
