# /tests/services/test_llm_generator.py
# unit test

import os
import tempfile
import pytest
import json

from app.services.llm_generator import CategoryManager

@pytest.fixture
def tempfile_path():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    yield path
    os.unlink(path)

@pytest.fixture
def temp_category_manager():
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(json.dumps({"categories": []}))
        tmp.flush()
        cm = CategoryManager(path=tmp.name)
    yield cm
    os.unlink(tmp.name)

# TC01: set_all

# TC01-1, 2: equivalent, edge case
@pytest.mark.parametrize("input_categories", [
    ["숙소", "음식"],     # 일반 케이스
    [],                  # 빈 리스트
    ["A"],               # 최소
    [f"항목{i}" for i in range(1000)]  # 긴 리스트
])
def test_set_all_valid_cases(temp_category_manager, input_categories):
    cm = temp_category_manager
    cm.set_all(input_categories)
    assert cm.get_all() == input_categories

# TC01-3: error
@pytest.mark.parametrize("bad_input", [None, 123, "숙소", {"카테고리": "숙소"}])
def test_set_all_invalid_input(temp_category_manager, bad_input):
    cm = temp_category_manager
    with pytest.raises(TypeError):
        cm.set_all(bad_input)

# TC01-4: specified
def test_set_all_output_shape_and_type(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소", "음식"])
    result = cm.get_all()
    assert isinstance(result, list)
    assert all(isinstance(i, str) for i in result)

# TC02: add


# TC02-1: equivalent
# TC02-2: edge case
@pytest.mark.parametrize("category", [
    "음식",        # 정상
    "",            # 빈 문자열
    "A" * 512      # 긴 문자열
])
def test_add_valid_input(temp_category_manager, category):
    cm = temp_category_manager
    cm.add(category)
    assert category in cm.get_all()
# TC02-3: error
@pytest.mark.parametrize("bad_input", [None, 123, ["숙소"]])
def test_add_invalid_input(temp_category_manager, bad_input):
    cm = temp_category_manager
    with pytest.raises(TypeError):
        cm.add(bad_input)

# TC02-4: specified
def test_add_no_duplicates(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소"])
    cm.add("숙소")
    assert cm.get_all().count("숙소") == 1

# TC03: remove

# TC03-1: equivalent
def test_remove_existing_and_nonexisting(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소", "음식"])
    cm.remove("음식")
    assert "음식" not in cm.get_all()

    cm.remove("없는값")  # 예외 없이 통과
    assert "없는값" not in cm.get_all()
# TC03-2: error
@pytest.mark.parametrize("bad_input", [None, 123, ["교통"]])
def test_remove_invalid_input(temp_category_manager, bad_input):
    cm = temp_category_manager
    with pytest.raises(TypeError):
        cm.remove(bad_input)
# TC03-3: specified
def test_remove_affects_output(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소", "음식"])
    cm.remove("숙소")
    assert "숙소" not in cm.get_all()
# TC04: get_all

# TC04-1: specified
def test_get_all_type(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소", "음식"])
    result = cm.get_all()
    assert isinstance(result, list)
    assert all(isinstance(i, str) for i in result)

# TC05: _load
@pytest.mark.parametrize("tmp_json", [
    "{잘못된 json:",  # JSONDecodeError
    "12345",          # TypeError (int에 .get() 못 씀)
])
def test_load_with_invalid_json(tmp_json):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(tmp_json)
        tmp.flush()
        cm = CategoryManager(tmp.name)
        assert cm.get_all() == []

    os.unlink(tmp.name)

# TC06: clear

# TC06-1: equivalent
def test_clear_removes_all_categories(temp_category_manager):
    cm = temp_category_manager
    cm.set_all(["숙소", "음식", "교통"])
    cm.clear()
    assert cm.get_all() == []

# TC06-2: edge case (이미 비어있을 때)
def test_clear_on_empty_state(temp_category_manager):
    cm = temp_category_manager
    assert cm.get_all() == []  # 초기 상태
    cm.clear()  # 실행 자체가 문제 없어야 함
    assert cm.get_all() == []

# TC06-3: specified (파일에 빈 리스트 저장됨 확인)
def test_clear_persists_empty_list_to_file(tempfile_path):
    cm = CategoryManager(tempfile_path)
    cm.set_all(["숙소"])
    cm.clear()

    # 파일 직접 열어서 확인
    with open(tempfile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data.get("categories") == []