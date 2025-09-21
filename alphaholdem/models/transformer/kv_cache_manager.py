"""KV Cache Manager for managing multiple player caches in poker transformer models."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class KVCacheManager:
    """Manages KV caches for multiple players in poker games."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the cache manager.

        Args:
            model: The transformer model (must have create_empty_cache method)
            device: Device to create caches on
        """
        self.model = model
        self.device = device
        self.player_caches: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def create_player_cache(self, player_id: int, batch_size: int) -> None:
        """Create an empty cache for a specific player."""
        if hasattr(self.model, "create_empty_cache"):
            self.player_caches[player_id] = self.model.create_empty_cache(
                batch_size, self.device
            )
        else:
            raise ValueError("Model must have create_empty_cache method")

    def get_player_cache(
        self, player_id: int
    ) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get the cache for a specific player."""
        return self.player_caches.get(player_id)

    def update_player_cache(
        self, player_id: int, new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Update the cache for a specific player."""
        self.player_caches[player_id] = new_cache

    def clear_player_cache(self, player_id: int) -> None:
        """Clear the cache for a specific player."""
        if player_id in self.player_caches:
            del self.player_caches[player_id]

    def clear_all_caches(self) -> None:
        """Clear all player caches."""
        self.player_caches.clear()

    def has_player_cache(self, player_id: int) -> bool:
        """Check if a player has a cache."""
        return player_id in self.player_caches

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current caches."""
        info = {
            "num_players": len(self.player_caches),
            "player_ids": list(self.player_caches.keys()),
            "cache_sizes": {},
        }

        for player_id, cache in self.player_caches.items():
            if cache:
                # Get size of first layer cache as representative
                first_layer_cache = next(iter(cache.values()))
                if first_layer_cache:
                    k_cache, v_cache = first_layer_cache
                    info["cache_sizes"][player_id] = {
                        "k_shape": list(k_cache.shape),
                        "v_shape": list(v_cache.shape),
                        "num_layers": len(cache),
                    }

        return info


class SelfPlayKVCacheManager:
    """Specialized KV cache manager for self-play scenarios."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the self-play cache manager.

        Args:
            model: The transformer model
            device: Device to create caches on
        """
        self.model = model
        self.device = device
        # Separate caches for self (player 0) and opponents
        self.self_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.opponent_caches: Dict[
            int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]
        ] = {}

    def initialize_self_cache(self, batch_size: int) -> None:
        """Initialize cache for self (player 0)."""
        if hasattr(self.model, "create_empty_cache"):
            self.self_cache = self.model.create_empty_cache(batch_size, self.device)
        else:
            raise ValueError("Model must have create_empty_cache method")

    def initialize_opponent_cache(self, opponent_id: int, batch_size: int) -> None:
        """Initialize cache for an opponent."""
        if hasattr(self.model, "create_empty_cache"):
            self.opponent_caches[opponent_id] = self.model.create_empty_cache(
                batch_size, self.device
            )
        else:
            raise ValueError("Model must have create_empty_cache method")

    def get_self_cache(self) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get the cache for self (player 0)."""
        return self.self_cache

    def get_opponent_cache(
        self, opponent_id: int
    ) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get the cache for a specific opponent."""
        return self.opponent_caches.get(opponent_id)

    def update_self_cache(
        self, new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Update the cache for self."""
        self.self_cache = new_cache

    def update_opponent_cache(
        self, opponent_id: int, new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Update the cache for a specific opponent."""
        self.opponent_caches[opponent_id] = new_cache

    def clear_self_cache(self) -> None:
        """Clear the cache for self."""
        self.self_cache = None

    def clear_opponent_cache(self, opponent_id: int) -> None:
        """Clear the cache for a specific opponent."""
        if opponent_id in self.opponent_caches:
            del self.opponent_caches[opponent_id]

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.self_cache = None
        self.opponent_caches.clear()

    def reset_for_new_game(self) -> None:
        """Reset all caches for a new game."""
        self.clear_all_caches()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current caches."""
        info = {
            "has_self_cache": self.self_cache is not None,
            "num_opponents": len(self.opponent_caches),
            "opponent_ids": list(self.opponent_caches.keys()),
            "cache_details": {},
        }

        if self.self_cache:
            info["cache_details"]["self"] = self._get_cache_size_info(self.self_cache)

        for opponent_id, cache in self.opponent_caches.items():
            info["cache_details"][f"opponent_{opponent_id}"] = (
                self._get_cache_size_info(cache)
            )

        return info

    def _get_cache_size_info(
        self, cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Get size information for a cache."""
        if not cache:
            return {"num_layers": 0}

        first_layer_cache = next(iter(cache.values()))
        if first_layer_cache:
            k_cache, v_cache = first_layer_cache
            return {
                "num_layers": len(cache),
                "k_shape": list(k_cache.shape),
                "v_shape": list(v_cache.shape),
            }
        return {"num_layers": len(cache)}
