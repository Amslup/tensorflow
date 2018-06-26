import socket
import struct

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes

class Message():
    def __init__(self, buf):
        self.buf = buf
        self.ptr = 0

    def read_byte(self): 
        return self.__read("b", 1)

    def read_short(self):
        return self.__read("h", 2)

    def read_int(self):
        return self.__read("i", 4)

    def read_long(self):
        return self.__read("q", 8)

    def skip(self, cnt):
        self.ptr += cnt

    def inner(self, length):
        b = self.buf[self.ptr:][:length]
        self.ptr += length
        return Message(b)

    def __read(self, t, l):
        b = self.buf[self.ptr:][:l]
        self.ptr += l
        return struct.unpack("<" + t, b)[0]

class TcpClient():
    def __init__(self, host, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = port
    
    def __enter__(self):
        self.s.connect((self.host, self.port))
        return self
    
    def __exit__(self, t, v, traceback):
        self.s.close()
    
    def write_byte(self, v):
        self.__write(v, "b")
        
    def write_short(self, v):
        self.__write(v, "h")
    
    def write_int(self, v):
        self.__write(v, "i")
        
    def write_long(self, v):
        self.__write(v, "q")
        
    def read_byte(self):
        return self.__read("b", 1)
        
    def read_short(self):
        return self.__read("h", 2)
        
    def read_int(self):
        return self.__read("i", 4)
        
    def read_long(self):
        return self.__read("q", 8)
    
    def read_data(self, l):
        b = self.s.recv(l)
        return Message(b)
        
    def __write(self, v, t):
        b = struct.pack("<" + t, v)
        self.s.sendall(b)
        
    def __read(self, t, l):
        b = self.s.recv(l)
        return struct.unpack("<" + t, b)[0]

class BinaryType():
    def __init__(self, type_id, type_name, fields):
        self.type_id = type_id
        self.type_name = type_name
        self.fields = fields

class BinaryField():
    def __init__(self, field_name, type_id, field_id):
        self.field_name = field_name
        self.type_id = type_id
        self.field_id = field_id
        
class TypeTreeNode():
    def __init__(self, name, type_id, fields=None):
        self.name = name
        self.type_id = type_id
        self.fields = fields

    def to_output_classes(self):
        if self.fields is None:
            return ops.Tensor
        else:
            res = {}
            for field in self.fields:
                res[field.name] = field.to_output_classes()
            return res

    def to_output_shapes(self):
        if self.fields is None:
            if self.type_id in range(1, 10):
                return tensor_shape.TensorShape([])
            elif self.type_id in range(12, 21):
                return tensor_shape.TensorShape([None])
            else:
                raise Exception("Unsupported type id " + str(self.type_id))
        else:
            res = {}
            for field in self.fields:
                res[field.name] = field.to_output_shapes()
            return res

    def to_output_types(self):
        if self.fields is None:
            if self.type_id in [1, 12]:
                return dtypes.int8
            elif self.type_id in [2, 13]:
                return dtypes.int16
            elif self.type_id in [3, 14]:
                return dtypes.int32
            elif self.type_id in [4, 15]:
                return dtypes.int64
            elif self.type_id in [5, 16]:
                return dtypes.float32
            elif self.type_id in [6, 17]:
                return dtypes.float64
            elif self.type_id in [7, 18]:
                return dtypes.uint16
            elif self.type_id in [8, 19]:
                return dtypes.bool
            elif self.type_id in [9, 20]:
                return dtypes.string
            else:
                raise Exception("Unsupported type id " + str(self.type_id))
        else:
            res = {}
            for field in self.fields:
                res[field.name] = field.to_output_types()
            return res  
        
    def to_flat(self):
        return self.__to_flat([])
    
    def __to_flat(self, res):
        res.append(self.type_id)
        if self.fields is not None:
            for field in self.fields:
                field.__to_flat(res)
        return res

class IgniteClient(TcpClient):
    def __init__(self, host, port):
        TcpClient.__init__(self, host, port)
        
    def handshake(self):
        self.write_int(8)    # Message length
        self.write_byte(1)   # Handshake operation
        self.write_short(1); # Version (1.0.0)
        self.write_short(0);
        self.write_short(0);
        self.write_byte(2);  # Thin client

        res_len = self.read_int()
        res = self.read_byte()

        if res != 1:
            raise Exception("Handshake failed")
            
    def get_cache_type(self, cache_name):
        cache_name_hash = self.__java_hash_code(cache_name)
        self.write_int(25)              # Message length
        self.write_short(2000)          # Operation code
        self.write_long(0)              # Request ID
        self.write_int(cache_name_hash) # Cache name
        self.write_byte(0)              # Flags
        self.write_byte(101)            # Filter (NULL)
        self.write_int(1)               # Cursor page size
        self.write_int(-1)              # Partition to query
        self.write_byte(0)              # Local flag

        result_length = self.read_int()
        request_id = self.read_long()
        status = self.read_int()
        
        if status != 0:
            raise Exception("Scan query status is " + str(status))

        cursor_id = self.read_long()
        row_count = self.read_int()
        
        if row_count == 0:
            raise Exception("Scan query returned empty result")

        payload = self.read_data(result_length - 25)

        self.read_byte()
        
        res = TypeTreeNode("root", 0, [
            self.__collect_types("key", payload), 
            self.__collect_types("val", payload)
        ])
        
        return res        
            
    def __java_hash_code(self, s):
        h = 0
        for c in s: 
            h = 31 * h + ord(c)
            h = h & 0xFFFFFFFF
        return h
    
    def __collect_types(self, field_name, data):
        type_id = data.read_byte()

        # Integer.
        if type_id == 3:
            data.skip(4)
            return TypeTreeNode(field_name, type_id)

        # Double Array.
        elif type_id == 17:
            length = data.read_int()
            data.skip(length * 8)
            return TypeTreeNode(field_name, type_id)

        # Wrapped Binary Object.
        elif type_id == 27:
            length = data.read_int()
            inner_data = data.inner(length)
            offset = data.read_int()
            return self.__collect_types(field_name, inner_data)

        # Complex Object.
        elif type_id == 103:
            obj_version = data.read_byte()
            obj_flags = data.read_short()
            obj_type_id = data.read_int()
            obj_hash_code = data.read_int()
            obj_length = data.read_int()
            obj_schema_id = data.read_int()
            obj_schema_offset = data.read_int()

            obj_type = self.__get_type(obj_type_id)
            children = []
            
            for obj_field in obj_type.fields:
                child = self.__collect_types(obj_field.field_name, data)
                children.append(child)

            data.skip(2)

            return TypeTreeNode(field_name, type_id, children)
        
        else:
            raise Exception("Unknown binary type " + str(type_id))
        
    def __get_type(self, type_id):
        self.write_int(14)      # Message length
        self.write_short(3002)  # Operation code
        self.write_long(0)      # Request ID
        self.write_int(type_id) # Type ID

        result_length = self.read_int()
        request_id = self.read_long()
        status = self.read_int()
        
        if status != 0:
            raise Exception("Get binary type status is " + str(status))

        binary_type_exists = self.read_byte()
        
        if binary_type_exists == 0:
            raise Exception("Binary type with id " + str(type_id) + " not found")
        
        binary_type_id = self.read_int()
        binary_type_name = self.__parse_string()
        affinity_field_name = self.__parse_string()

        fields = []
        for i in range(self.read_int()):
            field_name = self.__parse_string();
            field_type_id = self.read_int()
            field_id = self.read_int()
        
            field = BinaryField(field_name, field_type_id, field_id)
            fields.append(field)

        return BinaryType(binary_type_id, binary_type_name, fields)
    
    def __parse_string(self):
        header = self.read_byte()
        if header == 9:
            length = self.read_int()
            return self.read_data(length).buf.decode("utf-8")
        elif header == 101:
            return None
        else:
            raise Exception("Unknown binary type when expected string " + str(header))