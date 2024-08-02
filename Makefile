all:
	g++-11 example.cpp dbscan.cpp -I vendor/ -std=c++20 -o example
	./example

main:
	rm -rf main
	g++-11 main.cpp dbscan.cpp -I vendor/  -std=c++20 -o main
	./main &> junk.txt
test:
	g++-11 test_load_data.cpp -std=c++20 -o test_load_data
	./test_load_data

.PHONY:
	all