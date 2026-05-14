"""Custom optimizers για ezdl. Importing αυτό το package κάνει register
τους optimizers στο super-gradients OPTIMIZERS registry."""
from ezdl.optimizers.adopt import ADOPT  # noqa: F401  (triggers registration)
