model="EleutherAI/gpt-j-6B"
IFS='/'
read -ra newarr <<< "$model"
echo "${newarr[0]}"
p="models/models"
for r in ${newarr[@]}; do
    p+="--"
    p+=$r
done
echo $p
if [ -d "$p" ]; then
    echo "YEY"
fi