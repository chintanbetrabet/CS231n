
i=3
while [ $i -lt 4 ]
do
wkhtmltopdf http://cs231n.github.io/convolutional-networks// conv-neural-networks-$i.pdf
i=$((i+1))
done
