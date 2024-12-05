# Import Guard and Validator
from guardrails.hub import GibberishText
from guardrails.hub import NSFWText
from guardrails import Guard
def test_guardrails(query)->str:
    """
    Return None if the input is valid, otherwise return the error message.
    """
    # Use the Guard with the validator
    Gibberish_guard = Guard().use(
        GibberishText, threshold=0.5, validation_method="sentence", on_fail="exception"
    )
    try:
        # Test failing response
        Gibberish_guard.validate(query)
    except Exception as e:
        return "I'm sorry, I didn't quite understand that. Could you please rephrase or clarify your question?"
    
    NSFW_guard = Guard().use(
    NSFWText, threshold=0.8, validation_method="sentence", on_fail="exception")

    try:
        # Test failing response
        Gibberish_guard.validate(query)
        NSFW_guard.validate(query)
    except Exception as e:
        return "I'm sorry, but I cannot engage in that topic. Let's keep the conversation respectful and appropriate."
    return None