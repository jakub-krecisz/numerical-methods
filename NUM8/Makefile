DIR = $(notdir $(CURDIR))

a:
	python3 main.py point_a

b:
	python3 main.py point_b

clean:
	rm -rf __pycache__

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))

