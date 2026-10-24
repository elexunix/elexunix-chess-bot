.PHONY: compete debug sanitize sanitize_th main uci
CFLAGS = -g -std=c++26 -mavx2 -mfma
compete:
	g++ -Ofast $(CFLAGS) -Wunknown-pragmas compete.cpp -o compete
	nice -n20 ./compete $(ARGS) | tee -a logs
	rm compete
debug:
	g++ -O0 $(CFLAGS) main.cpp -o main
	g++ -O0 $(CFLAGS) compete.cpp -o compete
sanitize:
	g++ -fsanitize=address,undefined $(CFLAGS) compete.cpp -o compete
	g++ -fsanitize=address,undefined $(CFLAGS) main.cpp -o main
	./compete
sanitize_th:
	#g++ -g -fsanitize=thread,undefined $(CFLAGS) compete.cpp -o compete
	g++ -fsanitize=thread,undefined $(CFLAGS) main.cpp -o main
	./main
main:
	g++ -Ofast $(CFLAGS) main.cpp -o main
	#g++ -Ofast $(CFLAGS) uci.cpp -o uci
	./main
uci:
	g++ -Ofast $(CFLAGS) uci.cpp -o uci
dataset:
	g++ -Ofast $(CFLAGS) collect_dataset.cpp -o collect_dataset
	nice -n19 ./collect_dataset
	rm collect_dataset
	./analyze_dataset.py
