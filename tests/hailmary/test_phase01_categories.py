from __future__ import annotations

from hailmary.features import FeatureField, FeatureSchema
from hailmary.learning import RuleCondition


def test_rule_categories_are_exact_not_numeric_cluster_intervals() -> None:
    schema = FeatureSchema(
        "test.features.v2",
        (FeatureField("required_delay_s", "s"),),
        category_names=(
            "airport",
            "runway",
            "segment",
            "leader_cluster",
            "follower_cluster",
        ),
    )
    categories = {
        "airport": "KATL",
        "runway": "RW18R",
        "segment": "segment-common",
        "leader_cluster": "KATL:RW18R:1",
        "follower_cluster": "KATL:RW18R:2",
    }
    vector = schema.encode({"required_delay_s": 25.0}, categories=categories)
    condition = RuleCondition(
        "leader_follower",
        schema.schema_hash,
        {"required_delay_s": (10.0, 30.0)},
        {"segment": "segment-common", "follower_cluster": "KATL:RW18R:2"},
    )

    assert condition.matches(vector)
    assert not condition.matches(
        vector,
        categories={**categories, "follower_cluster": "KATL:RW18R:20"},
    )
    assert RuleCondition.from_dict(condition.to_dict()) == condition

