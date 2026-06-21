from __future__ import annotations

import math
import random
import unittest

import numpy as np

from game.cards import Carta
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy
from policy.linear_softmax_policy import add_scaled_in_place


def osservazione(
    mano: tuple[Carta, ...] = (
        Carta("coppe", "asso"),
        Carta("bastoni", "due"),
        Carta("denari", "tre"),
    ),
) -> Osservazione:
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


class TestLinearSoftmaxPolicy(unittest.TestCase):
    def test_initialize_crea_theta_della_dimensione_feature(self):
        # Il learner deve avere un parametro per ogni feature.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso", "carta_tre"])

        policy = LinearSoftmaxPolicy.initialize(extractor, rng=random.Random(0))

        self.assertEqual(len(policy.theta), extractor.size())
        self.assertEqual(policy.theta.dtype, np.float32)
        self.assertIs(policy.feature_extractor, extractor)

    def test_theta_con_dimensione_errata_solleva_value_error(self):
        # Evita bug silenziosi dove dot product e feature hanno lunghezze diverse.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])

        with self.assertRaises(ValueError):
            LinearSoftmaxPolicy(theta=[0.0, 1.0], feature_extractor=extractor)

    def test_probabilita_sommano_a_uno_sulle_carte_legali(self):
        # La softmax deve restituire una distribuzione solo sulle azioni legali.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(1),
        )

        probabilities = policy.action_probabilities(obs)

        self.assertEqual(set(probabilities), set(obs.azioni_legali))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-6)

    def test_softmax_resta_stabile_con_preferenze_grandi(self):
        # Sottrarre la preferenza massima evita overflow numerico.
        obs = osservazione(mano=(Carta("coppe", "asso"), Carta("bastoni", "due")))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[1000.0], feature_extractor=extractor)

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-6)
        self.assertTrue(all(math.isfinite(value) for value in probabilities.values()))

    def test_greedy_sceglie_la_carta_con_probabilita_massima(self):
        # In modalita' greedy la policy usa argmax invece del campionamento.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[2.0], feature_extractor=extractor)

        action = policy.select_action(obs, rng=random.Random(0), greedy=True)

        self.assertEqual(action, asso)

    def test_select_action_stocastico_restituisce_carta_legale(self):
        # In modalita' stocastica campiona dalla distribuzione sulle azioni legali.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(2),
        )

        action = policy.select_action(obs, rng=random.Random(3), greedy=False)

        self.assertIn(action, obs.azioni_legali)

    def test_log_probability_corrisponde_al_log_della_probabilita(self):
        # La log-probability serve poi al policy gradient.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        log_probability = policy.log_probability(obs, asso)

        self.assertAlmostEqual(log_probability, math.log(0.5))

    def test_grad_log_probability_ha_dimensione_theta_e_valore_atteso(self):
        # Per softmax lineare il gradiente e' feature azione meno feature attese.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        gradient = policy.grad_log_probability(obs, asso)

        self.assertEqual(len(gradient), len(policy.theta))
        self.assertAlmostEqual(gradient[0], 0.5)

    def test_action_non_legale_solleva_value_error_per_log_e_gradiente(self):
        # Log-probability e gradiente sono definiti solo su azioni legali.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(4),
        )
        illegal_action = Carta("spade", "asso")

        with self.assertRaises(ValueError):
            policy.log_probability(obs, illegal_action)

        with self.assertRaises(ValueError):
            policy.grad_log_probability(obs, illegal_action)

    def test_apply_gradient_aggiorna_theta(self):
        # L'update modifica i parametri nella direzione del gradiente.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        policy.apply_gradient([2.0], learning_rate=0.1)

        self.assertAlmostEqual(policy.theta[0], 0.2)

    # Il clipping resta supportato dal codice, ma non entra nel protocollo iniziale.
    # Lo riattiviamo come test quando scegliamo una soglia sperimentale motivata.
    # def test_apply_gradient_puo_clippare_la_norma(self):
    #     extractor = BriscolaFeatureExtractor(feature_names=["carta_asso", "carta_tre"])
    #     policy = LinearSoftmaxPolicy(theta=[0.0, 0.0], feature_extractor=extractor)
    #
    #     policy.apply_gradient([3.0, 4.0], learning_rate=1.0, max_update_norm=1.0)
    #
    #     self.assertAlmostEqual(policy.theta[0], 0.6)
    #     self.assertAlmostEqual(policy.theta[1], 0.8)

    def test_apply_gradient_con_dimensione_errata_solleva_value_error(self):
        # Il gradiente deve avere una componente per ogni parametro.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        with self.assertRaises(ValueError):
            policy.apply_gradient([1.0, 2.0], learning_rate=0.1)

    def test_copy_duplica_theta_ma_mantiene_feature_extractor(self):
        # Gli snapshot devono avere parametri indipendenti ma stessa codifica feature.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[1.0], feature_extractor=extractor)

        copied = policy.copy(name="snapshot")
        copied.theta[0] = 2.0

        self.assertEqual(copied.name, "snapshot")
        self.assertTrue(np.allclose(policy.theta, np.asarray([1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(copied.theta, np.asarray([2.0], dtype=np.float32)))
        self.assertEqual(copied.theta.dtype, np.float32)
        self.assertIs(copied.feature_extractor, extractor)

    def test_add_scaled_in_place_aggiorna_vettore_target(self):
        # Questo helper serve ad accumulare gradienti senza creare nuovi vettori.
        target = np.asarray([1.0, 2.0], dtype=np.float32)

        add_scaled_in_place(target, [3.0, 4.0], scale=0.5)

        self.assertTrue(np.allclose(target, np.asarray([2.5, 4.0], dtype=np.float32)))

    def test_mano_vuota_solleva_value_error(self):
        # Una distribuzione softmax richiede almeno una azione legale.
        obs = osservazione(mano=())
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(5),
        )

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
