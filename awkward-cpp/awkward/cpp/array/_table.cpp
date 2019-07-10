#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cinttypes>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <valarray>
#include "any.h"

namespace py = pybind11;

using T = std::int64_t;

// template <typename T>
class Table {
    py::object _view;
    std::shared_ptr<Table> _base;
    std::string rowname;
    ssize_t rowstart;
    std::map<std::string, std::valarray<T>> _contents;

   public:
    // Table.Row
    class Row {
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
                auto key = item.first.cast<std::string>();
                if (seen.find(key) != seen.end()) {
                    throw std::invalid_argument("column " + key +
                                                " occurs more than once");
                }
                seen.insert(key);
                auto value = item.second.cast<std::valarray<T>>();
                _contents.emplace(key, value);
            }
            if (py::len(columns2) != 0) {
                throw std::invalid_argument(
                    "only one positional argument when first argument is a "
                    "dict");
            }
        } else {
            ssize_t i = 0;
            auto value = columns1.cast<std::valarray<T>>();
            _contents.emplace(std::to_string(i++), value);
            for (auto item : columns2) {
                value = item.cast<std::valarray<T>>();
                _contents.emplace(std::to_string(i), value);
                seen.insert(std::to_string(i));
                i++;
            }
        }
        if (columns3) {
            for (auto item : columns3) {
                std::string key = item.first.cast<std::string>();
                if (seen.find(key) != seen.end()) {
                    throw std::invalid_argument("column " + key +
                                                " occurs more than once");
                }
                auto value = item.second.cast<std::valarray<T>>();
                _contents.emplace(key, value);
                seen.insert(key);
            }
        }
    }

    // No need to consider view anymore
    std::slice _index() {
        if (py::isinstance<py::none>(_view)) {
            return std::slice(0, 1, length());

        } else if (py::isinstance<py::tuple>(_view)) {
            auto myview = _view.cast<std::tuple<T,T,T>>();
            auto start = std::get<0>(myview);
            auto step = std::get<1>(myview);
            auto length = std::get<2>(myview);
            auto stop = start + step * length;
            // std::vector<T> ret(length / step + 1);
            // for (auto i = start; i < length; i += step) {
            //     ret.push_back(i);
            // }

            return std::slice(start, stop, step);

        } else {
            auto myview = _view.cast<ssize_t>();
            return std::slice(myview, 1, 1);
        }
    }

    ssize_t length() {
        ssize_t ret = 0;
        if (py::isinstance<py::none>(_view)) {
            if (_contents.size() != 0) {
                for (auto item : _contents) {
                    ret = std::min((size_t)ret, item.second.size());
                }
            }
        } else if (py::isinstance<py::tuple>(_view)) {
            auto myview = _view.cast<std::tuple<T,T,T>>();
            ret = std::get<2>(myview);
        } else {
            ret = _view.cast<ssize_t>();
        }
        return ret;
    }

    // void setitem(std::string where, py::handle whats) {
    //     if (!py::isinstance<py::none>(_view)) {
    //         throw std::domain_error(
    //             "new columns can only be attached to the original Table, not
    //             a " "view (try table.base['col'] = array");
    //     }
    // }

    std::valarray<T> getitem(std::string where) {
        // string, not string slice
        if (py::isinstance<py::none>(_view)) {
            if (_contents.find(where) == _contents.end()) {
                throw std::invalid_argument("no column named " + where);
            }
            return _contents[where];
        } else {
            auto index = _index();
            if (_contents.find(where) == _contents.end()) {
                throw std::invalid_argument("no column named " + where);
            }
            return _contents[where][index];
        }
    }

    // Row getitem(ssize_t where) {
    //     return Row(*this, where);
    // }

    // std::valarray<T> getitem(std::tuple<ssize_t> where) {
    //     if
    // }
};

PYBIND11_MODULE(_table, m) {
    py::class_<Table>(m, "Table")
        .def(py::init<py::object, py::args, py::kwargs>(),
             py::arg("columns3") = py::dict())
        .def("__len__", &Table::length)
        .def("__getitem__", &Table::getitem);

    py::class_<Table::Row>(m, "Table.Row")
        .def(py::init<const Table&, ssize_t>(), py::arg("table"),
             py::arg("index"));
}