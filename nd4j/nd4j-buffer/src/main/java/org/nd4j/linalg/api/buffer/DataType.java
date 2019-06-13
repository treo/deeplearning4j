/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.buffer;

public enum DataType {
    DOUBLE,
    FLOAT,
    HALF,
    LONG,
    INT,
    SHORT,
    UBYTE,
    BYTE,
    BOOL,
    UTF8,
    COMPRESSED,
    BFLOAT16,
    UINT16,
    UINT32,
    UINT64,
    UNKNOWN;


    public static DataType fromInt(int type) {
        switch (type) {
            case 1: return BOOL;
            case 3: return HALF;
            case 5: return FLOAT;
            case 6: return DOUBLE;
            case 7: return BYTE;
            case 8: return SHORT;
            case 9: return INT;
            case 10: return LONG;
            case 11: return UBYTE;
            case 12: return UINT16;
            case 13: return UINT32;
            case 14: return UINT64;
            case 17: return BFLOAT16;
            default: throw new UnsupportedOperationException("Unknown data type: [" + type + "]");
        }
    }

    public int toInt() {
        switch (this) {
            case BOOL: return 1;
            case HALF: return 3;
            case FLOAT: return 5;
            case DOUBLE: return 6;
            case BYTE: return 7;
            case SHORT: return 8;
            case INT: return 9;
            case LONG: return 10;
            case UBYTE: return 11;
            case UINT16: return 12;
            case UINT32: return 13;
            case UINT64: return 14;
            case BFLOAT16: return 17;
            case UTF8: return 50;
            default: throw new UnsupportedOperationException("Non-covered data type: [" + this + "]");
        }
    }

    /**
     * @return Returns true if the datatype is a floating point type (double, float or half precision)
     */
    public boolean isFPType(){
        return this == FLOAT || this == DOUBLE || this == HALF || this == BFLOAT16;
    }

    /**
     * @return Returns true if the datatype is an integer type (long, integer, short, ubyte or byte)
     */
    public boolean isIntType(){
        return this == LONG || this == INT || this == SHORT || this == UBYTE || this == BYTE || this == UINT16 || this == UINT32 || this == UINT64;
    }

    /**
     * @return True if the datatype is a numerical type and is signed (supports negative values)
     */
    public boolean isSigned(){
        switch (this){
            case DOUBLE:
            case FLOAT:
            case HALF:
            case LONG:
            case INT:
            case SHORT:
            case BYTE:
            case BFLOAT16:
                return true;
            case UBYTE:
            case BOOL:
            case UTF8:
            case COMPRESSED:
            case UINT16:
            case UINT32:
            case UINT64:
            case UNKNOWN:
            default:
                return false;
        }
    }

    /**
     * @return For fixed-width types, this returns the number of bytes per array element
     */
    public int width(){
        switch (this){
            case DOUBLE:
            case LONG:
            case UINT64:
                return 8;
            case FLOAT:
            case INT:
            case UINT32:
                return 4;
            case HALF:
            case SHORT:
            case BFLOAT16:
            case UINT16:
                return 2;
            case UBYTE:
            case BYTE:
            case BOOL:
                return 1;
            case UTF8:
            case COMPRESSED:
            case UNKNOWN:
            default:
                return -1;
        }
    }

    public static DataType fromNumpy(String numpyDtypeName){
        switch (numpyDtypeName.toLowerCase()){
            case "bool": return BOOL;
            case "byte": return BYTE;
            case "int8": return BYTE;
            case "int16": return SHORT;
            case "int32": return INT;
            case "int64": return LONG;
            case "uint8": return UBYTE;
            case "float16": return HALF;
            case "float32": return FLOAT;
            case "float64": return DOUBLE;
            case "uint16": return UINT16;
            case "uint32": return UINT32;
            case "uint64": return UINT64;
            case "complex64":
            case "complex128":
            case "complex_":
            default:
                throw new IllegalStateException("Unknown datatype or no ND4J equivalent datatype exists: " + numpyDtypeName);
        }
    }
}
