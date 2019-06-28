#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cinttypes>
#include <stdexcept>
#include <memory>

namespace py = pybind11;

class Table {
    class Row {
        // Table* _table;
        std::shared_ptr<Table> _table;
        ssize_t _index;

        Row(std::shared_ptr<Table> table, ssize_t index) {
            _table = table;
            _index = index;
        }
    };

    // class Table
    py::tuple _view;
    std::unique_ptr<Table> _base;
    std::string rowname;
    ssize_t rowstart;
    std::map<std::string, py::object> _contents;
    public:

    Table(py::dict columns1, py::args columns2, py::kwargs columns3) {
        _view = py::none();
        _base = NULL;
        rowname = "Row";
        rowstart = 0;
        
        std::set<std::string> seen;
        if (py::isinstance(columns1,py::dict())) {
            for (auto item : columns1) {
                if (seen.find(item.first.cast<std::string>()) != seen.end()){
                    throw std::invalid_argument("column "+item.first.cast<std::string>()+" occurs more than once");
                }
                seen.insert(item.first.cast<std::string>());

                // self[n] = x
                _contents.insert(item.first, item.second);
            }
            if (py::len(columns2) != 0) {
                throw std::invalid_argument("only one positional argument when first argument is a dict");
            }
        }
        else {
                // self["0"] = columns1
                _contents.insert("0",columns1);
                for (auto item : columns2){
                    _contents.insert(std::to_string(item.first.cast<std::int64_t>()+1),item.second);
                    seen.insert(std::to_string(columns2.first+1));
                }
        }
        
        for (auto item : columns3) {
            if (seen.find(item.first) != seen.end()) {
                throw std::invalid_argument("column "+item.first+" occurs more than once");
            }
            _contents.insert(std::to_string(columns2.first),item.second);
            seen.insert(item.first);
        }
        
    }
    
};


PYBIND11_MODULE(table, m) {
    
    py::class_<Table>(m, "Table")
        .def(py::init<py::dict, py::args, py::kwargs>(), py::arg("columns1"), py::arg("columns2"), py::arg("columns3"))
        // .def(py::init<std::string, py::args>(), py::arg("rowname"))
        // .def(py::init<py::numpy::recarray>(),py::arg("recarray"))
        // .def(py::init<std::pair<std::string,py::array>())
        // .def(py::init<py::tuple,Table>())
        // .def(py::init<py::tuple,std::unique_ptr<Table>>())
        .def("tolist",(py::list) &Table::tolist, "Returns a Pythonic representation")
        ;
    

    py::class_<Table::Row>(m,"Row")
        .def(py::init<std::shared_ptr<Table<T>>, ssize_t>(),py::arg("table"), py::arg("index"))

        ;
}