for f in A[A-Z].txt
do
 echo "Processing $f"
 cat $f | awk '{ print length, $0 }' | sort -n -s | cut -d" " -f2- > "sorted_$f"

done
