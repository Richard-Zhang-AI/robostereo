"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK


def get_current_action_mask(token_ids):
    # First filter to only action tokens
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    
    # Check if we're in training mode (has IGNORE_INDEX) or inference mode (no IGNORE_INDEX)
    has_ignore_index = (token_ids == IGNORE_INDEX).any()
    
    if has_ignore_index:
        # Training mode: count from first non-IGNORE_INDEX position
        newline_positions = token_ids != IGNORE_INDEX
        cumsum = torch.cumsum(newline_positions, dim=1)
    else:
        # Inference mode: count only action tokens
        cumsum = torch.cumsum(action_tokens_only_mask.long(), dim=1)
    
    # Total action tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
    total_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
    
    # Create the mask for all action tokens in the chunk
    mask = (1 <= cumsum) & (cumsum <= total_action_tokens)
    
    # Apply action token filter
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # First filter to only action tokens
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    
    # Check if we're in training mode (has IGNORE_INDEX) or inference mode (no IGNORE_INDEX)
    has_ignore_index = (token_ids == IGNORE_INDEX).any()
    
    if has_ignore_index:
        # Training mode: count from first non-IGNORE_INDEX position
        newline_positions = token_ids != IGNORE_INDEX
        cumsum = torch.cumsum(newline_positions, dim=1)
    else:
        # Inference mode: count only action tokens
        cumsum = torch.cumsum(action_tokens_only_mask.long(), dim=1)
    
    # For multi-step actions, total action tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
    total_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
    
    # Create the mask for tokens after the current action chunk
    mask = cumsum > total_action_tokens

    # Apply action token filter
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
