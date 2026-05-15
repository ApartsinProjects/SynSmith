"""Three LLM-based critics that drive the prompt update loop."""
from attrforge.critics.auditor import DiversityAuditor
from attrforge.critics.discriminator import RealismDiscriminator
from attrforge.critics.verifier import AttributeVerifier

__all__ = ["AttributeVerifier", "RealismDiscriminator", "DiversityAuditor"]
