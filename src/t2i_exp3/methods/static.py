"""Static5 reference using the controller's existing sequence execution path."""
from .feedback_routing_adapter import FeedbackRoutingMethod

STATIC5 = ["contextual_role_frame", "keyword_case_morph", "visual_euphemism_dictionary",
           "keyword_punctuation_split", "indirect_visual_proxy"]


def build(parameters, context):
    parameters = dict(parameters)
    budget = int(parameters.get("maximum_queries", 16))
    parameters["initial_query"] = "operator_sequence"
    parameters["initial_operator_sequence"] = STATIC5
    parameters["scheduled_initial_operator_sequences"] = {
        str(i): {"operator_sequence": STATIC5, "source": "origin"} for i in range(1, budget + 1)
    }
    return FeedbackRoutingMethod(parameters, context)
