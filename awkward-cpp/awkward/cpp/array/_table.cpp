#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cinttypes>
#include <stdexcept>
#include <memory>
#include "any.h"

namespace py = pybind11;

// template <typename T>
class Table {
    py::object _view;
    std::shared_ptr<Table> _base;
    std::string rowname;
    ssize_t rowstart;
    std::map<py::str, py::handle> _contents;

    ssize_t length() {
        return _contents.size();
    }
    
    // py::dict vs py::array????
    public: 
    class Row {
        // Table* _table;
        std::shared_ptr<Table> _table;
        ssize_t _index;

        public:
        Row(const Table& table, ssize_t index) {
            _table = std::make_shared<Table>(table);
            _index = index;
        }

        
    };

    // class Table
   

    Table(py::object columns1, py::args columns2, py::kwargs columns3) {
        _view = py::none();
        _base = NULL;
        rowname = "Row";
        rowstart = 0;
        
        std::set<std::string> seen;
        if (py::isinstance<py::dict>(columns1)) {
            auto columns = columns1.cast<py::dict>();
            for (auto item : columns) {
                std::string key = item.first.cast<std::string>();
                if (seen.find(key) != seen.end()){
                    throw std::invalid_argument("column "+key+" occurs more than once");
                }
                seen.insert(key);

           
                _contents.emplace(key, item.second);
            }
            if (py::len(columns2) != 0) {
                throw std::invalid_argument("only one positional argument when first argument is a dict");
            }
        }
        else {
                ssize_t i = 0;
                _contents.emplace(std::to_string(i++),columns1);
                for (auto item : columns2){
                    _contents.emplace(std::to_string(i),item);
                    seen.insert(std::to_string(i));
                    i++;
                }
        }
        if (columns3){
            for (auto item : columns3) {
                std::string key = item.first.cast<std::string>();
                if (seen.find(key) != seen.end()) {
                    throw std::invalid_argument("column "+key+" occurs more than once");
                }
                _contents.emplace(key,item.second);
                seen.insert(key);
            }
        }
        
    }   
    
};


PYBIND11_MODULE(_table, m) {
    
    py::class_<Table>(m, "Table")
        .def(py::init<py::object, py::args, py::kwargs>(), py::arg("columns3") = py::dict())
        ;
    

    py::class_<Table::Row>(m,"Table.Row")
        .def(py::init<const Table&, ssize_t>(),py::arg("table"), py::arg("index"))
        ;
}