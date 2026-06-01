import generators.models as models
import generators.prompts as prompts
import threading


class SimulatedUser:
    def __init__(self, config):
        self.config = config

        global_models = {
            "lock": threading.Lock(),
            "registered_models": {}
        }

        model_config_path = config.get("simulated_user_model_config")
        if not model_config_path:
            raise ValueError(
                "SimulatedUser requires 'simulated_user_model_config' in the run config; "
                "without it every scenario terminates on turn 1.")

        self.prompt_generator = prompts.get_generator(
            None,
            {"prompt_generator": "SimulatedUserPromptGenerator"},
            "SimulatedUserPromptGenerator"
        )

        try:
            self.model_generator = models.get_generator(
                global_models, model_config_path, None
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load simulated user model from {model_config_path}: {e}"
            ) from e

    def get_next_response(self, conversation_plan: str, history: list, last_agent_reply: str) -> str:
        payload = {
            "conversation_plan": conversation_plan,
            "history": history,
            "last_agent_reply": last_agent_reply
        }

        # Generate prompt
        self.prompt_generator.generate(payload)
        prompt = payload["prompt"]

        # Call model
        response = self.model_generator.generate(prompt)
        return response
