decodes=("slope")
for decode in "${decodes[@]}"; do
    bash run/fewshot/nli/layer_decoding_tests/rte_fewshot_hGivenP.sh $decode
    bash run/fewshot/nli/layer_decoding_tests/rte_fewshot.sh $decode
done