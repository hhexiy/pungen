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
	for split in train valid test; do \
		python scripts/make_src_tgt_files.py \
		-s data/roc/raw/$$split.txt \
		-k data/roc/raw/$$split.key \
		--num-key 2 \
		-o data/$(data)/$$split; \
	done

preprocess-txt-reddit:
	for split in train dev test; do \
		python scripts/make_src_tgt_files.py \
		--input Datasets/writingPrompts/valid.wp_target --data reddit \
		--output data/$(data) --filename $$split; \
	done

prepare-src-tgt-data:
	#for split in train; do \
	#	python scripts/extract_keywords.py -i data/onebillion/raw/parsed/$$split.txt -o data/$(data)/$$split --keywords NOUN VERB ADJ; \
	#done
		#-n 100 --debug;
	for split in train valid test; do \
		python scripts/make_src_tgt_files.py -i data/onebillion/raw/parsed/$$split.txt -o data/onebillion/$(data)/$$split -n 1000000; \
	done

fairseq-preprocess:
	python src/preprocess.py --source-lang src --target-lang tgt \
		--destdir data/$(data)/bin/data --thresholdtgt 20 --thresholdsrc 20 \
		--trainpref data/$(data)/train --validpref data/$(data)/valid \
		--model editor \
		#--testpref data/$(data)/test \
		--srcdict data/book/kw-sent/bin/data/dict.src.txt \
		--tgtdict data/book/kw-sent/bin/data/dict.tgt.txt \

skipgram-preprocess:
	python src/wordvec/preprocess.py --data-dir data/onebillion/wordvec \
		--corpus data/onebillion/raw/parsed/train.txt \
		--min-dist 5 --max-dist 10 --threshold 100 \
		--vocab data/onebillion/wordvec/dict.txt

train-skipgram:
	python src/wordvec/train.py --weights --cuda --data data/onebillion/wordvec/train.bin --save_dir models/onebillion/wordvec --mb 3500 --epoch 10 --vocab data/onebillion/wordvec/dict.txt

topk=10
generate-skipgram:
	python src/wordvec/generate.py --cuda --vocab data/$(data)/wordvec/dict.txt --model_path $(model) --pun-words semeval2017_task7/data/test/subtask3-heterographic-test.gold --puns data/semeval-pun/raw/hetero-pun.txt --output data/semeval-pun/kw-sent/gen-kw-1b.txt #--interact -k $(topk)

fusion=prob
insert=none
combine=embedding
train:
	python src/train.py data/$(data)/bin/data -a $(model) --source-lang src --target-lang tgt \
	--task edit --insert $(insert) --combine $(combine) \
	--criterion cross_entropy \
	--encoder lstm --decoder-attention True \
	--optimizer adagrad --lr 0.01 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --clip-norm 5 \
	--max-epoch 50 --max-tokens 6000 \
	--save-dir models/$(data)/$(ckpt) --no-progress-bar --log-interval 100 --no-epoch-checkpoints \
	#--pretrained-lm models/wikitext/wiki103.pt --mixing-weights learned --fusion-type $(fusion)

preprocess-test:
	python src/preprocess.py --source-lang src --target-lang tgt \
		--destdir data/$(data)/bin/data --thresholdtgt 20 --thresholdsrc 20 \
		--srcdict data/book/kw-sent/bin/data/dict.src.txt \
		--tgtdict data/book/kw-sent/bin/data/dict.tgt.txt \
		--testpref data/$(data)/gen-kw-lm

subset=test
test:
	python src/generate.py data/$(test_data)/bin/data --gen-subset $(subset) \
	--path models/$(model_data)/$(ckpt)/checkpoint_best.pt --beam 5 --nbest 5 --unkpen 100 \
	--sampling --sampling-temperature 0.3

interact:
	python src/interactive.py data/$(data)/bin/data \
		--path models/$(data)/$(ckpt)/checkpoint_best.pt \
		--beam 20 --nbest 20 --unkpen 100 --normal \
		#--sampling --sampling-temperature 0.2 \
		#--skipgram-model models/wordvec/sgns-e15.pt --skipgram-data data/wordvec \
		#--lm models/wikitext/wiki103.pt

lm-score:
	python src/lm_score.py data/$(data)/bin/data \
		--path models/$(data)/$(ckpt)/checkpoint_best.pt \
		--beam 10 --nbest 10 --unkpen 100 --normal \

analyze:
	python scripts/aggregate_results.py --model-outputs logs/$(data)/lstm.test.log logs/$(data)/lstm-wiki-input.test.log --model-names lstm lstm-wiki --output logs/$(data)/all.test.agg

human-eval:
	python scripts/human_eval.py --model-outputs logs/$(data)/lstm-wiki-input.test.log --num 1

retrieve:
	python src/retriever.py --doc-file data/$(data)/raw/sent.txt --lm-path models/wikitext --path models/retriever-1b.pkl --skipgram-path data/onebillion/wordvec/dict.txt models/onebillion/wordvec/sgns-e10.pt --keywords data/manual/pun.txt

system=rule
generate-pun:
	#python src/generate_pun.py --doc-file data/$(data)/raw/sent.txt --lm-path models/wikitext --retriever-path models/retriever-1b.pkl --skipgram-path data/onebillion/wordvec/dict.txt models/onebillion/wordvec/sgns-e10.pt --keywords data/manual/pun.txt
	python generate_pun.py data/$(data)/bin/data \
		--path models/$(data)/$(ckpt)/checkpoint_best.pt \
		--beam 20 --nbest 1 --unkpen 100 \
		--system $(system) \
		--doc-file data/onebillion/raw/sent.tokenized.ner.txt data/onebillion/raw/sent.tokenized.txt \
		--retriever-path models/retriever-1b.pkl \
		--lm-path models/wikitext --word-counts-path models/wikitext/dict.txt \
		--skipgram-path data/onebillion/wordvec/dict.txt models/onebillion/wordvec/sgns-e10.pt \
		--keywords data/manual/pun.txt

neural-generate:
	python src/generator.py data/$(data)/bin/data \
		--path models/$(data)/$(ckpt)/checkpoint_best.pt \
		--beam 50 --nbest 3 --unkpen 100 --insert $(insert)

retriever:
	python -m src.retriever --interactive --doc-file data/onebillion/raw/sent.tokenized.ner.txt data/onebillion/raw/sent.tokenized.txt --path models/onebillion/retriever.pkl --overwrite
