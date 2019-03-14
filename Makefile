data=roc/kw-story
src=title
tgt=story
glove=/u/scr/nlp/data/glove_vecs/glove.840B.300d.txt

skipgram-preprocess:
	python -m pungen.wordvec.preprocess --data-dir data/$(gdata)/skipgram \
		--corpus data/$(gdata)/raw/train.txt \
		--min-dist 5 --max-dist 10 --threshold 80 \
		--vocab data/$(gdata)/skipgram/dict.txt

train-skipgram:
	python -m pungen.wordvec.train --weights --cuda --data data/$(gdata)/skipgram/train.bin --save_dir models/$(gdata)/skipgram --mb 3500 --epoch 15 --vocab data/$(gdata)/skipgram/dict.txt

topk=10
generate-skipgram:
	python -m pungen.wordvec.generate --cuda \
		--skipgram-model data/$(gdata)/skipgram/dict.txt models/$(gdata)/skipgram/sgns-e15.pt \
		--pun-words data/semeval/hetero/dev.json \
		--output results/semeval/hetero/dev.topic_words.json \
		--logfile results/semeval/hetero/dev.topic_words.log \
		--cuda -k 50

fusion=prob
insert=deleted
combine=token
train:
	python -m pungen.train data/$(gdata)/$(data)/bin/data -a $(model) --source-lang src --target-lang tgt \
	--task edit --insert $(insert) --combine $(combine) \
	--criterion cross_entropy \
	--encoder lstm --decoder-attention True \
	--optimizer adagrad --lr 0.01 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --clip-norm 5 \
	--max-epoch 50 --max-tokens 7000 \
	--save-dir models/$(gdata)/$(data)/$(ckpt) --no-progress-bar --log-interval 5000 \
	#--pretrained-lm models/wikitext/wiki103.pt --mixing-weights learned --fusion-type $(fusion)

analyze:
	python scripts/aggregate_results.py --model-outputs logs/$(data)/lstm.test.log logs/$(data)/lstm-wiki-input.test.log --model-names lstm lstm-wiki --output logs/$(data)/all.test.agg

human-eval:
	python scripts/human_eval.py --model-outputs logs/$(data)/lstm-wiki-input.test.log --num 1

system=rule
gdata=bookcorpus
split=dev
generate-pun:
	python generate_pun.py data/$(gdata)/$(data)/bin/data \
		--path models/$(gdata)/$(data)/$(ckpt)/checkpoint_best.pt \
		--beam 20 --nbest 1 --unkpen 100 \
		--system $(system) --task edit \
		--retriever-model models/$(gdata)/retriever.pkl --doc-file data/$(gdata)/raw/sent.tokenized.txt \
		--lm-path models/wikitext/wiki103.pt --word-counts-path models/wikitext/dict.txt \
		--skipgram-model data/$(gdata)/skipgram/dict.txt models/$(gdata)/skipgram/sgns-e15.pt \
		--num-candidates 500 --num-templates 100 \
		--num-topic-word 100 --type-consistency-threshold 0.3 \
		--pun-words data/semeval/hetero/$(split).json \
		--outdir results/semeval/hetero/$(split)/$(outdir) \
		--scorer random \
		--learned-scorer-weights results/score-eval/lr_model.pkl \
		--learned-scorer-features results/score-eval/features.pkl \
		--max-num-examples 100

semeval_dir=semeval2017_task7/data/test/subtask3-
type=hetero
parse-semeval:
	PYTHONPATH=. python scripts/parse_semeval.py --xml $(semeval_dir)$(type)graphic-test.xml --gold $(semeval_dir)$(type)graphic-test.gold --output data/semeval/$(type)

# Process generic corpus
split-file:
	split -l 1000000 data/$(gdata)/raw/sent.tokenized.txt data/$(gdata)/raw/parts/x
## Parse in parallel
# bash scripts/submit_preprocess.sh
## Join files
# ls --color=no * | xargs cat > ...

# Parsed to tokenized sentences
parsed-to-tokens:
	PYTHONPATH=. python scripts/parsed_to_tokenized.py --input data/$(gdata)/raw/sent.tokenized.parsed.txt --output data/$(gdata)/raw/sent.tokenized.txt

build-retriever:
	python -m pungen.retriever --doc-file data/$(gdata)/raw/sent.tokenized.txt --path models/$(gdata)/retriever.pkl --overwrite


human-corr:
	python eval_scoring_func.py --human-eval data/funniness_annotation/$(data)_pun_scores.txt \
		--lm-path models/wikitext/wiki103.pt --word-counts-path models/wikitext/dict.txt \
	--skipgram-model data/$(gdata)/skipgram/dict.txt models/$(gdata)/skipgram/sgns-e15.pt --outdir results/pun-analysis/$(data) \
	--features grammar ratio ambiguity distinctiveness --analysis --ignore-cache  
	#--features grammar ratio ambiguity distinctiveness --analysis --ignore-cache  

prepare-pun-data:
	PYTHONPATH=. python scripts/make_pun_src_tgt_files.py --pun-data data/semeval/$(type)/dev.json --output data/pun/ --dev-frac 0.1

split-data:
	python scripts/split.py -i data/$(gdata)/raw/sent.tokenized.parsed.txt -o data/$(gdata)/raw --split 0.9 0.1 --split-names train valid --shuffle

prepare-src-tgt-data:
	set -e;
	for split in train valid; do \
		PYTHONPATH=. python scripts/make_src_tgt_files.py -i data/$(gdata)/raw/$$split.txt -o data/$(gdata)/$(data)/$$split --delete-frac 0.5 --window-size 2 --random-window-size; \
	done

fairseq-preprocess:
	python -m pungen.preprocess --source-lang src --target-lang tgt \
		--destdir data/$(gdata)/$(data)/bin/data --thresholdtgt 80 --thresholdsrc 80 \
		--validpref data/$(gdata)/$(data)/valid \
		--trainpref data/$(gdata)/$(data)/train \
		--workers 8
		#--srcdict data/$(gdata)/$(data)/bin/data/dict.src.txt \
		#--tgtdict data/$(gdata)/$(data)/bin/data/dict.tgt.txt
		#--trainpref data/$(gdata)/$(data)/train 
		#--testpref data/$(data)/test \
		#--tgtdict data/book/kw-sent/bin/data/dict.tgt.txt \

fairseq-preprocess-lm:
	python -m pungen.preprocess --only-source \
		--destdir data/$(gdata)/$(data)/bin/data --thresholdtgt 80 --thresholdsrc 80 \
		--validpref data/$(gdata)/$(data)/valid.txt \
		--trainpref data/$(gdata)/$(data)/train.txt \
		--workers 8

train-lm:
	python -m pungen.train --task language_modeling data/$(gdata)/$(data)/bin/data \
	  --max-epoch 35 --arch lstm_lm --optimizer adagrad \
	  --lr 0.01 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --clip-norm 5 \
	  --criterion adaptive_loss \
	  --decoder-embed-dim 1024 --decoder-layers 2 --decoder-attention False \
	  --adaptive-softmax-cutoff 10000,20000,40000 --max-tokens 6000 --tokens-per-sample 1024 --ddp-backend no_c10d

all-results:
	python scripts/aggregate_results.py --output-dirs results/semeval/hetero/retrieve results/semeval/hetero/retrieve+swap results/semeval/hetero/rule results/semeval/hetero/rule+neural \
		--names retrieve retrieve+swap rule rule+neural \
		--output results/semeval/hetero/all.txt

render-results:
	python scripts/render_results.py --input results/semeval/hetero/$(outdir)/results.json

to-nanyun:
	PYTHONPATH=. python scripts/process_results/to_nanyun.py --result-dir results/semeval/hetero/test/batch2 --outdir results/semeval/hetero/test/batch2/to_nanyun
