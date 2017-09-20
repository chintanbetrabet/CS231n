i=4
while [ $i -lt 10 ]
do
i=$((i+1))
wget http://cs231n.stanford.edu/slides/2016/winter1516_lecture$i.pdf
done
