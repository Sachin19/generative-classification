decodes=("slope")
for decode in "${decodes[@]}"; do
    bash run/fewshot/nli/layer_decoding_tests/sick_fewshot_hGivenP.sh $decode
    bash run/fewshot/nli/layer_decoding_tests/sick_fewshot.sh $decode
done