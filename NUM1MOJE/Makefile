DIR = $(notdir $(CURDIR))

plot:
	python3 main.py plot

table:
	python3 main.py table

setup: requirements.txt
	pip3 install -r requirements.txt

clean:
	rm -rf __pycache__ *.png

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))
