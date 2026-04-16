def extract_assistant_template(tokenizer):
    """Extract response template from tokenizer's chat template

    Supports both 'assistant' role (standard) and 'model' role (Gemma).
    """

    # Try assistant role first (standard), then model role (Gemma)
    for assistant_role in ["assistant", "model"]:
        sample_messages = [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": assistant_role, "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        try:
            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                sample_messages, tokenize=False, add_generation_prompt=False
            )

            # Find where assistant content starts
            assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
            if assistant_start < 0:
                continue

            # Find where the user content ends
            user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
            if user_start < 0:
                continue
            user_end = user_start + len("__USER_PLACEHOLDER__")

            return formatted[user_end:assistant_start]
        except Exception:
            continue

    # If we get here, neither role worked
    raise ValueError("Could not extract assistant template - neither 'assistant' nor 'model' role worked")


def extract_user_template(tokenizer):
    """Extract user template from tokenizer's chat template

    Supports both 'assistant' role (standard) and 'model' role (Gemma).
    """

    # Try assistant role first (standard), then model role (Gemma)
    for assistant_role in ["assistant", "model"]:
        sample_messages = [
            {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": assistant_role, "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        try:
            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                sample_messages, tokenize=False, add_generation_prompt=False
            )

            # Find where user content starts
            user_start = formatted.find("__USER_PLACEHOLDER__")
            if user_start < 0:
                continue

            # Find where the system content ends
            system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
            if system_start < 0:
                continue
            system_end = system_start + len("__SYSTEM_PLACEHOLDER__")

            return formatted[system_end:user_start]
        except Exception:
            continue

    # If we get here, neither role worked
    raise ValueError("Could not extract user template - neither 'assistant' nor 'model' role worked")
