DIR = $(notdir $(CURDIR))

a_show:
	python3 main.py point_a show

a_save:
	python3 main.py point_a save

b_show:
	python3 main.py point_b show

b_save:
	python3 main.py point_b save

clean:
	rm -rf __pycache__

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))

