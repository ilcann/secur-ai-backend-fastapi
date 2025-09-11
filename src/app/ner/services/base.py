from abc import ABC, abstractmethod
from typing import List, Optional
from app.ner.schemas import NerEntity
from app.ner.schemas import EntityLabelDto

class NerBaseService(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> List[NerEntity]:
        """
        Verilen metin için entity listesi döndürür
        """
        pass

    def update_labels(self, labels: List[str]):
        """
        Verilen etiketleri günceller
        """
        pass

    def get_labels(self) -> Optional[List[str]]:
        """
        Mevcut etiketleri döndürür
        """
        pass

    async def fetch_labels(self) -> Optional[List[EntityLabelDto]]:
        """
        Entity etiketlerini fetch eder
        """
        pass

    def extract_keys(self, labels: Optional[List[EntityLabelDto]]) -> List[str]:
        """
        Entity etiketlerinden anahtarları çıkarır
        """
        if not labels:
            return []
        return [label.key for label in labels]