"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings.yaml"
"""
import argparse

from myconfig.settings import Settings

from myevaluation.model_evaluation_mAP import SparseObjectDetModel
from myevaluation.model_evaluation_mAP import DenseObjectDetModel
from myevaluation.model_evaluation_mAP import FBSparseVGGModel
from myevaluation.model_evaluation_mAP import SparseRecurrentObjectDetModel



def main():
    parser = argparse.ArgumentParser(description='Evaluate network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file

    settings = Settings(settings_filepath, generate_log=True)
    
    if settings.model_name == 'fb_sparse_vgg':
        evaluator = FBSparseVGGModel(settings)
    elif settings.model_name == 'dense_vgg':
        evaluator = DenseVGGModel(settings)
    elif (settings.model_name == 'sparse_REDnet' or 
         settings.model_name == 'custom_sparse_REDnetv1' or 
         settings.model_name == 'sparse_firenet'):
        evaluator = SparseRecurrentObjectDetModel(settings)
    elif settings.model_name == 'fb_sparse_object_det':
        evaluator = SparseObjectDetModel(settings)
    elif settings.model_name == 'dense_object_det':
        evaluator = DenseObjectDetModel(settings)
    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()
