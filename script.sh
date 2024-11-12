kernels=(2 3 4 5)
N=(512 1024 4096)

# ./run 512 512 512 100 6 0

for kernel in ${kernels[@]}; do
	for n in ${N[@]}; do
		echo "Running kernel $kernel with N=$n"
		./run $n $n $n 1009 $kernel 0
	done
	
	echo ""
done