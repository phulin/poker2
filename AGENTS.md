# AGENTS.md
This is a python project. The main entry points are src/alphaholdem/cli/{train_rebel,train_kbest}.py.

## Virtual environment
* Before running any Python commands, source the venv with `source venv/bin/activate`.
* Run python commands with `python`. Don't use `chmod +x`.

## General instructions
* We try to keep our patches minimal, writing the smallest amount of code that successfully achieves our goals.
* Avoid "defensive programming" - don't add excessive try/catch or checks on incoming data. Do, however, add internal consistency checks to data structures when appropriate.
* All new functionality should come with a complete test.
* After making a change, please make sure that it works either via passing the relevant tests or by a custom script that gives confidence that it works.
* When writing a one-off debugging script, run it to make sure it works.
* If making major changes, run the whole test suite with `pytest`.
* When asked to identify an issue, don't immediately jump to writing code unless another part of the prompt tells you to. Just give the user an explanation of what the issue is.

## Tests
* When writing tests, try to make the test short and clear, and share relevant code between tests. The user should be able to read the test and quickly understand exactly what behavior the test is verifying.
* Use `torch.testing.assertclose` as your assertion where you can. Don't adjust the rtol/atol.