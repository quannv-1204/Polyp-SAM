from typing import Dict, Any

from .polypSAM import PolypSAM
from .module.sub_head import SubHead

class AutomaticPolypSAM(PolypSAM):
    """ This returns low_res_masks and iou_predictions (see `forward`)
    to use for next iterative forward
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super(AutomaticPolypSAM, self).__init__(*args, **kwargs)

        self.sub_head = SubHead() 

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = True):
        """
        This forward function does not accept batch input
        """

        image = input.get("image")

        image_embedding = self.image_encoder(image[None])

        sub_mask = self.sub_head(image_embedding.detach())

        points = input.get("point_prompt")
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=input.get("box_prompt", None),
            masks=input.get("mask_input", None),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions, sub_mask
