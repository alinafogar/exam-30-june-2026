"""Decision logs for frozen Briscola policy games."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.environment import Ambiente, CartaPescata, EsitoMossa, EventoPubblico
from game.observation import Osservazione
from game.rules import NUMERO_GIOCATORI, valida_giocatore_id
from policy import Policy


@dataclass(frozen=True)
class DecisionOutcome:
    """Public outcome immediately after a recorded decision."""

    partita_finita: bool
    presa_completata: bool
    carte_presa_completata: tuple[CartaGiocata, ...]
    vincitore_presa: int | None
    punti_presa: int
    carte_pescate: tuple[CartaPescata, ...]
    prossimo_giocatore: int | None
    punteggi: dict[str, int]
    eventi_pubblici: tuple[EventoPubblico, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """One legal observation, chosen action, and public outcome."""

    step_index: int
    giocatore_id: int
    policy_name: str
    osservazione: Osservazione
    azioni_legali: tuple[Carta, ...]
    action_probabilities: dict[Carta, float]
    azione: Carta
    greedy: bool
    outcome: DecisionOutcome


@dataclass(frozen=True)
class DecisionLog:
    """Complete diagnostic log for one played game."""

    seed_ambiente: int
    seed_policy: int
    primo_giocatore_id: int
    greedy: bool
    records: tuple[DecisionRecord, ...]
    punteggi_finali: dict[str, int]
    squadra_vincitrice: str | None


def record_decision_log(
    *,
    policies_by_player: dict[int, Policy],
    seed_ambiente: int,
    seed_policy: int,
    primo_giocatore_id: int,
    greedy: bool,
) -> DecisionLog:
    """Play one game and record only legal observations plus public outcomes."""

    _valida_policies_by_player(policies_by_player)
    valida_giocatore_id(primo_giocatore_id)

    ambiente = Ambiente(
        seed=seed_ambiente,
        primo_giocatore_id=primo_giocatore_id,
    )
    rng_policy = random.Random(seed_policy)
    records: list[DecisionRecord] = []

    while not ambiente.finita:
        step_index = len(records)
        giocatore_id = ambiente.giocatore_corrente
        osservazione = ambiente.osserva(giocatore_id)
        azioni_legali = osservazione.azioni_legali
        policy = policies_by_player[giocatore_id]
        action_probabilities = policy.action_probabilities(osservazione)
        _valida_probabilities(action_probabilities, azioni_legali)

        azione = policy.select_action(osservazione, rng_policy, greedy=greedy)
        if azione not in azioni_legali:
            raise ValueError("La policy ha scelto una carta non legale")

        esito = ambiente.gioca(azione)
        records.append(
            DecisionRecord(
                step_index=step_index,
                giocatore_id=giocatore_id,
                policy_name=policy.name,
                osservazione=osservazione,
                azioni_legali=azioni_legali,
                action_probabilities=dict(action_probabilities),
                azione=azione,
                greedy=greedy,
                outcome=_decision_outcome(esito),
            )
        )

    ambiente.verifica_integrita_stato()

    return DecisionLog(
        seed_ambiente=seed_ambiente,
        seed_policy=seed_policy,
        primo_giocatore_id=primo_giocatore_id,
        greedy=greedy,
        records=tuple(records),
        punteggi_finali=dict(ambiente.punteggi),
        squadra_vincitrice=ambiente.squadra_vincitrice(),
    )


def _decision_outcome(esito: EsitoMossa) -> DecisionOutcome:
    return DecisionOutcome(
        partita_finita=esito.partita_finita,
        presa_completata=esito.presa_completata,
        carte_presa_completata=esito.carte_presa_completata,
        vincitore_presa=esito.vincitore_presa,
        punti_presa=esito.punti_presa,
        carte_pescate=esito.carte_pescate,
        prossimo_giocatore=esito.prossimo_giocatore,
        punteggi=dict(esito.punteggi),
        eventi_pubblici=esito.eventi_pubblici,
    )


def _valida_policies_by_player(policies_by_player: dict[int, Policy]) -> None:
    expected_players = set(range(NUMERO_GIOCATORI))
    actual_players = set(policies_by_player)
    if actual_players != expected_players:
        raise ValueError(
            "policies_by_player deve contenere esattamente i giocatori "
            f"{sorted(expected_players)}, ottenuto {sorted(actual_players)}"
        )
    for giocatore_id in policies_by_player:
        valida_giocatore_id(giocatore_id)


def _valida_probabilities(
    action_probabilities: dict[Carta, float],
    azioni_legali: tuple[Carta, ...],
) -> None:
    if set(action_probabilities) != set(azioni_legali):
        raise ValueError(
            "action_probabilities deve avere esattamente le azioni legali"
        )
