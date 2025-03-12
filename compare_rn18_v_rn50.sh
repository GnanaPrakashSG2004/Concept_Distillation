
python extract_model_activations.py --model resnet18.a2_in1k --dataset_split val --feature_layer_version v1 --output_root /scratch/$(whoami)/rsvc-exps/ --start_class_idx 0 --end_class_idx 100
python extract_model_activations.py --model resnet50.a2_in1k --dataset_split val --feature_layer_version v1 --output_root /scratch/$(whoami)/rsvc-exps/ --start_class_idx 0 --end_class_idx 100

python extract_concepts.py --model resnet18.a2_in1k --dataset_split val --feature_layer_version v1 --output_root /scratch/$(whoami)/rsvc-exps/ --start_class_idx 0 --end_class_idx 100
python extract_concepts.py --model resnet50.a2_in1k --dataset_split val --feature_layer_version v1 --output_root /scratch/$(whoami)/rsvc-exps/ --start_class_idx 0 --end_class_idx 100

# python compare_models.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
# --start_class_idx 0 --end_class_idx 100 --folder_exists overwrite \
# --concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/

python compare_models.py --dataset_split_0 val --dataset_split_1 val --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json --start_class_idx 0 --end_class_idx 100 --folder_exists overwrite --concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/

python evaluate_regression.py --dataset_split_0 val --dataset_split_1 val --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/ \
--start_class_idx 0 --end_class_idx 100 \
--data_split val --patchify

python concept_integrated_gradients.py --dataset_split_0 val --dataset_split_1 val --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--model_0 resnet18.a2_in1k --model_1 resnet50.a2_in1k \
--concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/

python replacement_test.py --dataset_split_0 val --dataset_split_1 val --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/

python visualize_similarity_vs_importance.py --dataset_split_0 val --dataset_split_1 val --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--concept_root_folder_0 /scratch/$(whoami)/rsvc-exps/ --concept_root_folder_1 /scratch/$(whoami)/rsvc-exps/
