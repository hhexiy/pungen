keywords:
	python pytorch_src/generate.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.test \
		--checkpoint models/ROCstory_title_keywords_e500_h1000_edr0.4_hdr0.1.pt \
		--task cond_generate \
		--conditional-data rocstory_plan_write/ROCStories_all_merge_tokenize.title.test \
		--cuda --temperature 0.15 --sents 100 --dedup \
		--outf generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt

story:
	python pytorch_src/generate.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test \
		--checkpoint models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt \
		--task cond_generate \
		--conditional-data generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt \
		--cuda --temperature 0.3 \
		--outf generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt_lm_e1000_h1500_edr0.2_hdr0.1_t0.3.txt

models/story_dict.pkl:
	python pytorch_src/make_vocab.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test \
		--output models/story_dict.pkl

models/keyword_dict.pkl:
	python pytorch_src/make_vocab.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.test \
		--output models/keyword_dict.pkl

interact2:
	python pytorch_src/interactive_generate.py \
		--keyword-model models/ROCstory_title_keywords_e500_h1000_edr0.4_hdr0.1.pt \
		--story-model models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt \
		--keyword-vocab models/keyword_dict.pkl \
		--story-vocab models/story_dict.pkl \
		--titles rocstory_plan_write/ROCStories_all_merge_tokenize.title.test \
		--cuda --temperature 0.3

data=roc/kw-story
src=title
tgt=story
preprocess-txt-roc:
	for split in train dev test; do \
		python scripts/make_src_tgt_files.py \
		--input rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.$$split \
		--src $(src) --tgt $(tgt) \
		--join --sep " <eot> " \
		--output data/$(data) --filename $$split; \
	done

preprocess-txt-reddit:
	for split in train dev test; do \
		python scripts/make_src_tgt_files.py \
		--input Datasets/writingPrompts/valid.wp_target --data reddit \
		--output data/$(data) --filename $$split; \
	done

preprocess-pt:
	python src/preprocess.py -train_src data/$(data)/src_train.txt -train_tgt data/$(data)/tgt_train.txt -valid_src data/$(data)/src_dev.txt -valid_tgt data/$(data)/tgt_dev.txt -save_data data/$(data)/data #-dynamic_dict

preprocess-fairseq:
	python src/preprocess.py --source-lang src --target-lang tgt \
		--trainpref data/$(data)/train --validpref data/$(data)/valid --testpref data/$(data)/test \
		--destdir data/$(data)/bin/data --thresholdtgt 10 --thresholdsrc 10

fusion=prob
train:
	python src/train.py data/$(data)/bin/data -a $(model) --source-lang src --target-lang tgt \
	--criterion cross_entropy \
	--decoder-attention True \
	--lr 0.05 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --clip-norm 5 \
	--max-epoch 50 --max-tokens 3000 \
	--save-dir models/$(data)/$(ckpt) --no-progress-bar --log-interval 100 --no-epoch-checkpoints \
	#--pretrained-lm models/wikitext/wiki103.pt --mixing-weights learned --fusion-type $(fusion)

test:
	python src/generate.py data/$(data)/bin/data --gen-subset valid \
	--path models/$(data)/$(ckpt)/checkpoint_best.pt --beam 10 --unkpen 100 #--max-examples -1 #--no-score \
	#--unkpen 1 --sampling --sampling-temperature 0.8 --sampling-topk 5 

interact:
	python src/interactive.py data/$(data)/bin/data \
		--path models/$(data)/$(model)/checkpoint_best.pt \
		--beam 20 --nbest 1 --unkpen 100 \
		#--lm models/wikitext/wiki103.pt

gluon-preprocess:
	python src/preprocess.py --train data/$(data)/train.txt --outdir data/$(data)

gpu=0
arch=lstm
model=lm
src=keywords
tgt=story
gluon-train:
	export MXNET_FORCE_ADDTAKEGRAD=1; export MXNET_GPU_MEM_POOL_TYPE=Round; \
	python src/train.py --train data/$(data)/train.txt --valid data/$(data)/dev.txt \
	--src $(src) --tgt $(tgt) --vocab data/$(data)/$(model)-$(src)-$(tgt)-vocab.json \
	--batch-size 128 --num-buckets 10 \
	--optimizer sgd --lr 0.1 --lr-update-factor 0.5 --clip-norm 5 --min-lr 1e-5 --max-epoch 100 \
	--model-type $(model) -a $(arch) \
	--log-interval 100 --save-dir checkpoints/$(data)/$(ckpt) \
	--gpu $(gpu) --seed 2 

gluon-gen:
	python src/generate.py --test data/$(data)/train.txt --batch-size 10 --checkpoint checkpoints/$(data)/$(ckpt)/best --output out --prefix-len 30 --alpha 0 -k 5 --max-len 100 --beam 100 --gpu 0 --sampling --temperature 0.5
