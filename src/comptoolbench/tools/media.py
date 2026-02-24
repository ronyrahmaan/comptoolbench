"""Media tools: image generation and audio transcription.

These tools simulate media operations with deterministic outputs.
Real media APIs would require paid services, so both modes return
simulated results for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


def _sim_hash(seed: str, *args: Any) -> int:
    """Deterministic hash for simulated data."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16)


@register_tool
class GenerateImage(BaseTool):
    """Generate an image from a text prompt (simulated)."""

    name = "generate_image"
    schema = ToolSchema(
        name="generate_image",
        description="Generate an image based on a text description/prompt. Returns an image URL and metadata about the generated image.",
        category=ToolCategory.MEDIA,
        parameters=[
            ToolParameter(
                name="prompt",
                type="string",
                description="Text description of the image to generate (e.g. 'a sunset over mountains')",
            ),
            ToolParameter(
                name="size",
                type="string",
                description="Image size",
                enum=["256x256", "512x512", "1024x1024"],
                default="512x512",
                required=False,
            ),
            ToolParameter(
                name="style",
                type="string",
                description="Image style",
                enum=["photorealistic", "illustration", "abstract", "sketch"],
                default="photorealistic",
                required=False,
            ),
        ],
        returns="Generated image URL and metadata",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # No free image generation API — always simulated
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        prompt = kwargs["prompt"]
        size = kwargs.get("size", "512x512")
        style = kwargs.get("style", "photorealistic")
        h = _sim_hash("image", prompt, size, style)

        return {
            "status": "generated",
            "image_url": f"https://images.example.com/generated/{h % 1000000:06d}.png",
            "prompt": prompt,
            "size": size,
            "style": style,
            "generation_id": f"img_{h % 10000000:07d}",
            "estimated_cost_usd": 0.02 if size == "256x256" else 0.04 if size == "512x512" else 0.08,
        }


@register_tool
class TranscribeAudio(BaseTool):
    """Transcribe audio to text (simulated)."""

    name = "transcribe_audio"
    schema = ToolSchema(
        name="transcribe_audio",
        description="Transcribe an audio file or URL to text. Returns the transcription with timestamps and confidence scores.",
        category=ToolCategory.MEDIA,
        parameters=[
            ToolParameter(
                name="audio_source",
                type="string",
                description="URL or path to the audio file to transcribe",
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Language of the audio (e.g. 'en', 'es', 'ja')",
                default="en",
                required=False,
            ),
        ],
        returns="Transcription text with metadata",
        returns_type="object",
    )

    # Pre-defined transcriptions for reproducibility
    _TRANSCRIPTIONS: dict[str, dict[str, Any]] = {
        "meeting": {
            "text": "Good morning everyone. Let's start with the quarterly review. Revenue is up 15 percent compared to last quarter. Our customer satisfaction scores have improved to 4.6 out of 5. The main challenge remains scaling our infrastructure to meet demand.",
            "duration_seconds": 22,
            "segments": [
                {"start": 0.0, "end": 3.5, "text": "Good morning everyone."},
                {"start": 3.5, "end": 7.2, "text": "Let's start with the quarterly review."},
                {"start": 7.2, "end": 12.8, "text": "Revenue is up 15 percent compared to last quarter."},
                {"start": 12.8, "end": 17.1, "text": "Our customer satisfaction scores have improved to 4.6 out of 5."},
                {"start": 17.1, "end": 22.0, "text": "The main challenge remains scaling our infrastructure to meet demand."},
            ],
        },
        "lecture": {
            "text": "Today we'll discuss neural network architectures. Transformers have become the dominant architecture for natural language processing. The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input.",
            "duration_seconds": 18,
            "segments": [
                {"start": 0.0, "end": 4.0, "text": "Today we'll discuss neural network architectures."},
                {"start": 4.0, "end": 10.5, "text": "Transformers have become the dominant architecture for natural language processing."},
                {"start": 10.5, "end": 18.0, "text": "The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input."},
            ],
        },
        "voicemail": {
            "text": "Hi, this is Sarah from the marketing team. I wanted to follow up on the campaign proposal we discussed on Tuesday. Could you send me the updated budget numbers by end of day Friday? Thanks, talk to you soon.",
            "duration_seconds": 14,
            "segments": [
                {"start": 0.0, "end": 3.8, "text": "Hi, this is Sarah from the marketing team."},
                {"start": 3.8, "end": 8.5, "text": "I wanted to follow up on the campaign proposal we discussed on Tuesday."},
                {"start": 8.5, "end": 12.5, "text": "Could you send me the updated budget numbers by end of day Friday?"},
                {"start": 12.5, "end": 14.0, "text": "Thanks, talk to you soon."},
            ],
        },
    }

    def execute_live(self, **kwargs: Any) -> Any:
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        audio_source = kwargs["audio_source"]
        language = kwargs.get("language", "en")
        h = _sim_hash("transcribe", audio_source)

        # Try to match a predefined transcription
        for keyword, data in self._TRANSCRIPTIONS.items():
            if keyword in audio_source.lower():
                return {
                    "status": "completed",
                    "audio_source": audio_source,
                    "language": language,
                    "text": data["text"],
                    "duration_seconds": data["duration_seconds"],
                    "confidence": 0.95,
                    "segments": data["segments"],
                    "word_count": len(data["text"].split()),
                }

        # Generic transcription for unknown sources
        return {
            "status": "completed",
            "audio_source": audio_source,
            "language": language,
            "text": "This is a transcription of the provided audio. The speaker discusses various topics including project updates, timeline adjustments, and next steps for the team.",
            "duration_seconds": 30 + (h % 120),
            "confidence": round(0.85 + (h % 15) / 100, 2),
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "This is a transcription of the provided audio."},
                {"start": 15.0, "end": 30.0, "text": "The speaker discusses various topics including project updates, timeline adjustments, and next steps for the team."},
            ],
            "word_count": 26,
        }
