.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf result tmp/

.gitignore:
	printf '.env\nresult\ntmp/\n' > $@

flake.nix:
	nix flake init

main.py:
	touch $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.lock flake.nix main.py
	nix develop --ignore-environment .#all
	touch $@

tmp/check-done: .gitignore flake.lock flake.nix main.py
	nix develop --ignore-environment .#check
	touch $@
