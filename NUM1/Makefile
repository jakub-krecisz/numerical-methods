DIR = $(notdir $(CURDIR))

plot:
	python3 main.py plot

table_float:
	python3 main.py table float32

table_double:
	python3 main.py table double

clean:
	rm -rf __pycache__ *.png

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))
