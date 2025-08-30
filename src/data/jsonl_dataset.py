import json
from typing import Dict, List

from torch.utils.data import Dataset


class JsonlTextDataset(Dataset):
	"""Simple JSONL dataset that returns dict(text=...)."""

	def __init__(self, path: str, text_field: str = "text") -> None:
		self._samples: List[str] = []
		self._text_field = text_field
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				obj = json.loads(line)
				text = obj.get(text_field)
				if not isinstance(text, str):
					# Skip malformed rows silently; could alternatively raise
					continue
				self._samples.append(text)

	def __len__(self) -> int:
		return len(self._samples)

	def __getitem__(self, idx: int) -> Dict[str, str]:
		return {"text": self._samples[idx]} 