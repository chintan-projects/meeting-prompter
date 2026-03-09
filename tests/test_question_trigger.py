"""Tests for question trigger — scoring, rhetorical suppression (F-201)."""

from lib.triggers.question_trigger import (
    score_question,
    _is_tag_question,
    _is_self_answered,
    _is_rhetorical,
)


# ─── Genuine questions (must still score high) ───────────────────────────


class TestGenuineQuestions:
    """Real information-seeking questions that MUST trigger."""

    def test_direct_pricing_question(self) -> None:
        assert score_question("What is the pricing for enterprise customers?") >= 0.5

    def test_api_capability(self) -> None:
        assert score_question("Does the API support batch processing?") >= 0.5

    def test_deployment_question(self) -> None:
        assert score_question("How does the deployment pipeline handle rollbacks?") >= 0.5

    def test_integration_question(self) -> None:
        assert score_question("Can you integrate with our existing Salesforce setup?") >= 0.5

    def test_security_question(self) -> None:
        assert score_question("What security certifications does your platform have?") >= 0.5

    def test_performance_question(self) -> None:
        assert score_question("What is the average latency for API responses?") >= 0.5

    def test_comparison_question(self) -> None:
        assert score_question("What is the difference between the starter and pro plans?") >= 0.5

    def test_feature_question(self) -> None:
        assert score_question("Do you support single sign-on for enterprise accounts?") >= 0.5

    def test_plain_question_mark(self) -> None:
        assert score_question("Is there a way to export the full audit log?") >= 0.25

    def test_how_question_without_mark(self) -> None:
        assert score_question("How would we handle the migration from the old system") >= 0.25


# ─── Tag question suppression ────────────────────────────────────────────


class TestTagQuestionSuppression:
    """Statements with trailing confirmation tags must score zero."""

    def test_right_tag(self) -> None:
        assert score_question("We should ship this by Friday, right?") == 0.0

    def test_okay_tag(self) -> None:
        assert score_question("The deadline is next week, okay?") == 0.0

    def test_yeah_tag(self) -> None:
        assert score_question("We can handle that volume, yeah?") == 0.0

    def test_isnt_it_tag(self) -> None:
        assert score_question("The new design looks much better, isn't it?") == 0.0

    def test_dont_you_think_tag(self) -> None:
        assert score_question("This approach seems reasonable, don't you think?") == 0.0

    def test_isnt_that_right_tag(self) -> None:
        assert score_question("We agreed on the quarterly release, isn't that right?") == 0.0

    def test_you_know_tag(self) -> None:
        assert score_question("The performance has been solid lately, you know?") == 0.0


# ─── Self-answering suppression ──────────────────────────────────────────


class TestSelfAnsweringSuppression:
    """Questions immediately followed by self-answers must score zero."""

    def test_yeah_self_answer(self) -> None:
        assert score_question("Can we do that? Yeah I think we can handle it.") == 0.0

    def test_i_think_so(self) -> None:
        assert score_question("Is that the right approach? I think so actually.") == 0.0

    def test_probably(self) -> None:
        assert score_question("Will they accept that timeline? Probably not a big deal.") == 0.0

    def test_yes_self_answer(self) -> None:
        assert score_question("Does that make sense? Yes it should work fine.") == 0.0

    def test_of_course(self) -> None:
        assert score_question("Can we ship on time? Of course we just need to finalize.") == 0.0

    def test_well_self_answer(self) -> None:
        assert score_question("Is this the best option? Well we could also consider plan B.") == 0.0


# ─── Rhetorical question suppression ─────────────────────────────────────


class TestRhetoricalSuppression:
    """Rhetorical question forms must score zero."""

    def test_dont_we_already(self) -> None:
        assert score_question("Don't we already have this feature in the product?") == 0.0

    def test_isnt_it_obvious(self) -> None:
        assert score_question("Isn't it obvious that we need more resources here?") == 0.0

    def test_why_would_anyone(self) -> None:
        assert score_question("Why would anyone use the old deployment process?") == 0.0

    def test_who_cares(self) -> None:
        assert score_question("Who cares about the legacy integration at this point?") == 0.0

    def test_whats_the_point(self) -> None:
        assert score_question("What's the point of maintaining backward compatibility?") == 0.0

    def test_do_we_really_need(self) -> None:
        assert score_question("Do we really need another round of testing?") == 0.0


# ─── Edge cases — genuine questions resembling rhetorical forms ──────────


class TestEdgeCases:
    """Boundary: genuine questions that resemble suppressed forms must NOT be suppressed."""

    def test_genuine_do_we_need(self) -> None:
        """'Do we need X?' without 'really' is genuine."""
        assert score_question("Do we need additional infrastructure for scaling?") >= 0.25

    def test_genuine_does_it(self) -> None:
        """Plain auxiliary question is genuine."""
        assert score_question("Does it support real-time data synchronization?") >= 0.25

    def test_genuine_is_there(self) -> None:
        assert score_question("Is there a limit on the number of API calls?") >= 0.25

    def test_genuine_can_we(self) -> None:
        assert score_question("Can we get a demo of the enterprise dashboard?") >= 0.25

    def test_question_with_right_in_content(self) -> None:
        """'right' as content word, not tag, should still score."""
        assert score_question("What is the right approach for handling rate limits?") >= 0.25

    def test_short_genuine_not_tag(self) -> None:
        """A genuine question ending with a question mark (no comma-tag)."""
        assert score_question("Is that something your platform supports natively?") >= 0.25


# ─── Unit tests for suppression helper functions ─────────────────────────


class TestHelperFunctions:
    """Direct tests for _is_tag_question, _is_self_answered, _is_rhetorical."""

    def test_tag_detection_positive(self) -> None:
        assert _is_tag_question("we should do this, right?") is True
        assert _is_tag_question("the api is stable, isn't it?") is True
        assert _is_tag_question("looks good, okay?") is True

    def test_tag_detection_negative(self) -> None:
        assert _is_tag_question("what is the right approach?") is False
        assert _is_tag_question("is that okay for the timeline?") is False

    def test_self_answer_positive(self) -> None:
        assert _is_self_answered("can we? yeah definitely.") is True
        assert _is_self_answered("is it ready? i think so.") is True

    def test_self_answer_negative(self) -> None:
        assert _is_self_answered("what is the pricing?") is False
        assert _is_self_answered("no question mark here") is False
        assert _is_self_answered("can we get a demo?") is False

    def test_rhetorical_positive(self) -> None:
        assert _is_rhetorical("don't we already have that?") is True
        assert _is_rhetorical("why would anyone do that?") is True
        assert _is_rhetorical("do we really need this?") is True

    def test_rhetorical_negative(self) -> None:
        assert _is_rhetorical("do we need more servers?") is False
        assert _is_rhetorical("what does the api return?") is False


# ─── Existing behavior preserved ─────────────────────────────────────────


class TestExistingFilters:
    """Existing pre-scoring filters still work correctly."""

    def test_too_short(self) -> None:
        assert score_question("is it?") == 0.0

    def test_incomplete_ending(self) -> None:
        assert score_question("what about the integration with") == 0.0

    def test_fragment_pattern(self) -> None:
        assert score_question("can you tell me?") == 0.0

    def test_empty_string(self) -> None:
        assert score_question("") == 0.0

    def test_score_capped_at_one(self) -> None:
        result = score_question(
            "What is the pricing for this integration API? How does it compare?"
        )
        assert result <= 1.0
