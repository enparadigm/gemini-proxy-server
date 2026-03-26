from __future__ import annotations

from loguru import logger


class UsageTracker:
    TEXT_INPUT_RATE = 0.50
    AUDIO_INPUT_RATE = 3.00
    TEXT_OUTPUT_RATE = 2.00
    AUDIO_OUTPUT_RATE = 12.00

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[dict] = []

    def record_turn(self, usage: dict):
        input_audio = 0
        input_text = 0
        output_audio = 0
        output_text = 0

        for item in usage.get("promptTokensDetails", []):
            if item.get("modality") == "AUDIO":
                input_audio += item.get("tokenCount", 0)
            elif item.get("modality") == "TEXT":
                input_text += item.get("tokenCount", 0)

        for item in usage.get("candidatesTokensDetails", []):
            if item.get("modality") == "AUDIO":
                output_audio += item.get("tokenCount", 0)
            elif item.get("modality") == "TEXT":
                output_text += item.get("tokenCount", 0)

        self.turns.append(
            {
                "prompt_token_count": usage.get("promptTokenCount", 0),
                "candidates_token_count": usage.get("candidatesTokenCount", 0),
                "input_audio_token_count": input_audio,
                "input_text_token_count": input_text,
                "output_audio_token_count": output_audio,
                "output_text_token_count": output_text,
            }
        )

    def calculate_cost(self) -> dict:
        if not self.turns:
            return {}

        # All fields are cumulative per turn — sum across all turns for billing
        total_input_audio = sum(t["input_audio_token_count"] for t in self.turns)
        total_input_text = sum(t["input_text_token_count"] for t in self.turns)
        total_output_audio = sum(t["output_audio_token_count"] for t in self.turns)
        total_output_text = sum(t["output_text_token_count"] for t in self.turns)

        has_modality = total_input_audio > 0 or total_input_text > 0

        if has_modality:
            input_audio_cost = (total_input_audio / 1_000_000) * self.AUDIO_INPUT_RATE
            input_text_cost = (total_input_text / 1_000_000) * self.TEXT_INPUT_RATE
            output_audio_cost = (
                total_output_audio / 1_000_000
            ) * self.AUDIO_OUTPUT_RATE
            output_text_cost = (total_output_text / 1_000_000) * self.TEXT_OUTPUT_RATE
        else:
            # Fallback: no modality breakdown, treat everything as audio
            total_input = sum(t["prompt_token_count"] for t in self.turns)
            total_output = sum(t["candidates_token_count"] for t in self.turns)
            input_audio_cost = (total_input / 1_000_000) * self.AUDIO_INPUT_RATE
            input_text_cost = 0.0
            output_audio_cost = (total_output / 1_000_000) * self.AUDIO_OUTPUT_RATE
            output_text_cost = 0.0

        total_cost = (
            input_audio_cost + input_text_cost + output_audio_cost + output_text_cost
        )

        return {
            "session_id": self.session_id,
            "total_input_audio": total_input_audio,
            "total_input_text": total_input_text,
            "total_output_audio": total_output_audio,
            "total_output_text": total_output_text,
            "input_audio_cost": input_audio_cost,
            "input_text_cost": input_text_cost,
            "output_audio_cost": output_audio_cost,
            "output_text_cost": output_text_cost,
            "total_cost_usd": total_cost,
            "total_cost_inr": total_cost * 94,
        }

    def print_report(self):
        cost = self.calculate_cost()
        logger.info(
            "\n{}\n"
            "💰 USAGE REPORT — session: {}\n"
            "{}\n"
            "  Turns with usage data : {}\n"
            "  ── Input ──\n"
            "  Audio tokens          : {:,}\n"
            "  Text tokens           : {:,}\n"
            "  ── Output ──\n"
            "  Audio tokens          : {:,}\n"
            "  Text tokens           : {:,}\n"
            "  ── Costs ──\n"
            "  Audio input           : ${:.6f}\n"
            "  Text input            : ${:.6f}\n"
            "  Audio output          : ${:.6f}\n"
            "  Text output           : ${:.6f}\n"
            "  ── Total ──\n"
            "  USD                   : ${:.6f}\n"
            "  INR                   : ₹{:.4f}\n"
            "{}\n",
            "=" * 60,
            self.session_id,
            "=" * 60,
            len(self.turns),
            cost.get("total_input_audio", 0),
            cost.get("total_input_text", 0),
            cost.get("total_output_audio", 0),
            cost.get("total_output_text", 0),
            cost.get("input_audio_cost", 0.0),
            cost.get("input_text_cost", 0.0),
            cost.get("output_audio_cost", 0.0),
            cost.get("output_text_cost", 0.0),
            cost.get("total_cost_usd", 0.0),
            cost.get("total_cost_inr", 0.0),
            "=" * 60,
        )

