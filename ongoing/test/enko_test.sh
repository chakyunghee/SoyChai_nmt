MODEL_FN=$1
GPU_ID=0
BEAM_SIZE=5
TEST_FN=./test.tok.bpe.head-1000.en
REF_TEST_FN=./test.tok.bpe.head-1000.detok.tok.ko

cat ${TEST_FN} | python ../translate.py --model ${MODEL_FN} --gpu_id ${GPU_ID} --lang enko --beam_size ${BEAM_SIZE} | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ${REF_TEST_FN}
