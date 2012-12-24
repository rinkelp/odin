/**********************************************************************
 * cbf_copy.c -- cbflib copy functions                                *
 *                                                                    *
 * Version 0.9.1 23 February 2010                                     *
 *                                                                    *
 * (C) Copyright 2010 Herbert J. Bernstein                            *
 *                                                                    *
 *                      Part of the CBFlib API                        *
 *                              by                                    *
 *                          Paul Ellis and                            *
 *         Herbert J. Bernstein (yaya@bernstein-plus-sons.com)        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

/**********************************************************************
 *                                                                    *
 * YOU MAY REDISTRIBUTE THE CBFLIB PACKAGE UNDER THE TERMS OF THE GPL *
 *                                                                    *
 * ALTERNATIVELY YOU MAY REDISTRIBUTE THE CBFLIB API UNDER THE TERMS  *
 * OF THE LGPL                                                        *
 *                                                                    *
 **********************************************************************/

/*************************** GPL NOTICES ******************************
 *                                                                    *
 * This program is free software; you can redistribute it and/or      *
 * modify it under the terms of the GNU General Public License as     *
 * published by the Free Software Foundation; either version 2 of     *
 * (the License, or (at your option) any later version.               *
 *                                                                    *
 * This program is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
 * GNU General Public License for more details.                       *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this program; if not, write to the Free Software        *
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA           *
 * 02111-1307  USA                                                    *
 *                                                                    *
 **********************************************************************/

/************************* LGPL NOTICES *******************************
 *                                                                    *
 * This library is free software; you can redistribute it and/or      *
 * modify it under the terms of the GNU Lesser General Public         *
 * License as published by the Free Software Foundation; either       *
 * version 2.1 of the License, or (at your option) any later version. *
 *                                                                    *
 * This library is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU  *
 * Lesser General Public License for more details.                    *
 *                                                                    *
 * You should have received a copy of the GNU Lesser General Public   *
 * License along with this library; if not, write to the Free         *
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,    *
 * MA  02110-1301  USA                                                *
 *                                                                    *
 **********************************************************************/

/**********************************************************************
 *                                                                    *
 *                    Stanford University Notices                     *
 *  for the CBFlib software package that incorporates SLAC software   *
 *                 on which copyright is disclaimed                   *
 *                                                                    *
 * This software                                                      *
 * -------------                                                      *
 * The term ‘this software’, as used in these Notices, refers to      *
 * those portions of the software package CBFlib that were created by *
 * employees of the Stanford Linear Accelerator Center, Stanford      *
 * University.                                                        *
 *                                                                    *
 * Stanford disclaimer of copyright                                   *
 * --------------------------------                                   *
 * Stanford University, owner of the copyright, hereby disclaims its  *
 * copyright and all other rights in this software.  Hence, anyone    *
 * may freely use it for any purpose without restriction.             *
 *                                                                    *
 * Acknowledgement of sponsorship                                     *
 * ------------------------------                                     *
 * This software was produced by the Stanford Linear Accelerator      *
 * Center, Stanford University, under Contract DE-AC03-76SFO0515 with *
 * the Department of Energy.                                          *
 *                                                                    *
 * Government disclaimer of liability                                 *
 * ----------------------------------                                 *
 * Neither the United States nor the United States Department of      *
 * Energy, nor any of their employees, makes any warranty, express or *
 * implied, or assumes any legal liability or responsibility for the  *
 * accuracy, completeness, or usefulness of any data, apparatus,      *
 * product, or process disclosed, or represents that its use would    *
 * not infringe privately owned rights.                               *
 *                                                                    *
 * Stanford disclaimer of liability                                   *
 * --------------------------------                                   *
 * Stanford University makes no representations or warranties,        *
 * express or implied, nor assumes any liability for the use of this  *
 * software.                                                          *
 *                                                                    *
 * Maintenance of notices                                             *
 * ----------------------                                             *
 * In the interest of clarity regarding the origin and status of this *
 * software, this and all the preceding Stanford University notices   *
 * are to remain affixed to any copy or derivative of this software   *
 * made or distributed by the recipient and are to be affixed to any  *
 * copy of software made or distributed by the recipient that         *
 * contains a copy or derivative of this software.                    *
 *                                                                    *
 * Based on SLAC Software Notices, Set 4                              *
 * OTT.002a, 2004 FEB 03                                              *
 **********************************************************************/



/**********************************************************************
 *                               NOTICE                               *
 * Creative endeavors depend on the lively exchange of ideas. There   *
 * are laws and customs which establish rights and responsibilities   *
 * for authors and the users of what authors create.  This notice     *
 * is not intended to prevent you from using the software and         *
 * documents in this package, but to ensure that there are no         *
 * misunderstandings about terms and conditions of such use.          *
 *                                                                    *
 * Please read the following notice carefully.  If you do not         *
 * understand any portion of this notice, please seek appropriate     *
 * professional legal advice before making use of the software and    *
 * documents included in this software package.  In addition to       *
 * whatever other steps you may be obliged to take to respect the     *
 * intellectual property rights of the various parties involved, if   *
 * you do make use of the software and documents in this package,     *
 * please give credit where credit is due by citing this package,     *
 * its authors and the URL or other source from which you obtained    *
 * it, or equivalent primary references in the literature with the    *
 * same authors.                                                      *
 *                                                                    *
 * Some of the software and documents included within this software   *
 * package are the intellectual property of various parties, and      *
 * placement in this package does not in any way imply that any       *
 * such rights have in any way been waived or diminished.             *
 *                                                                    *
 * With respect to any software or documents for which a copyright    *
 * exists, ALL RIGHTS ARE RESERVED TO THE OWNERS OF SUCH COPYRIGHT.   *
 *                                                                    *
 * Even though the authors of the various documents and software      *
 * found here have made a good faith effort to ensure that the        *
 * documents are correct and that the software performs according     *
 * to its documentation, and we would greatly appreciate hearing of   *
 * any problems you may encounter, the programs and documents any     *
 * files created by the programs are provided **AS IS** without any   *
 * warranty as to correctness, merchantability or fitness for any     *
 * particular or general use.                                         *
 *                                                                    *
 * THE RESPONSIBILITY FOR ANY ADVERSE CONSEQUENCES FROM THE USE OF    *
 * PROGRAMS OR DOCUMENTS OR ANY FILE OR FILES CREATED BY USE OF THE   *
 * PROGRAMS OR DOCUMENTS LIES SOLELY WITH THE USERS OF THE PROGRAMS   *
 * OR DOCUMENTS OR FILE OR FILES AND NOT WITH AUTHORS OF THE          *
 * PROGRAMS OR DOCUMENTS.                                             *
 **********************************************************************/

/**********************************************************************
 *                                                                    *
 *                           The IUCr Policy                          *
 *      for the Protection and the Promotion of the STAR File and     *
 *     CIF Standards for Exchanging and Archiving Electronic Data     *
 *                                                                    *
 * Overview                                                           *
 *                                                                    *
 * The Crystallographic Information File (CIF)[1] is a standard for   *
 * information interchange promulgated by the International Union of  *
 * Crystallography (IUCr). CIF (Hall, Allen & Brown, 1991) is the     *
 * recommended method for submitting publications to Acta             *
 * Crystallographica Section C and reports of crystal structure       *
 * determinations to other sections of Acta Crystallographica         *
 * and many other journals. The syntax of a CIF is a subset of the    *
 * more general STAR File[2] format. The CIF and STAR File approaches *
 * are used increasingly in the structural sciences for data exchange *
 * and archiving, and are having a significant influence on these     *
 * activities in other fields.                                        *
 *                                                                    *
 * Statement of intent                                                *
 *                                                                    *
 * The IUCr's interest in the STAR File is as a general data          *
 * interchange standard for science, and its interest in the CIF,     *
 * a conformant derivative of the STAR File, is as a concise data     *
 * exchange and archival standard for crystallography and structural  *
 * science.                                                           *
 *                                                                    *
 * Protection of the standards                                        *
 *                                                                    *
 * To protect the STAR File and the CIF as standards for              *
 * interchanging and archiving electronic data, the IUCr, on behalf   *
 * of the scientific community,                                       *
 *                                                                    *
 * * holds the copyrights on the standards themselves,                *
 *                                                                    *
 * * owns the associated trademarks and service marks, and            *
 *                                                                    *
 * * holds a patent on the STAR File.                                 *
 *                                                                    *
 * These intellectual property rights relate solely to the            *
 * interchange formats, not to the data contained therein, nor to     *
 * the software used in the generation, access or manipulation of     *
 * the data.                                                          *
 *                                                                    *
 * Promotion of the standards                                         *
 *                                                                    *
 * The sole requirement that the IUCr, in its protective role,        *
 * imposes on software purporting to process STAR File or CIF data    *
 * is that the following conditions be met prior to sale or           *
 * distribution.                                                      *
 *                                                                    *
 * * Software claiming to read files written to either the STAR       *
 * File or the CIF standard must be able to extract the pertinent     *
 * data from a file conformant to the STAR File syntax, or the CIF    *
 * syntax, respectively.                                              *
 *                                                                    *
 * * Software claiming to write files in either the STAR File, or     *
 * the CIF, standard must produce files that are conformant to the    *
 * STAR File syntax, or the CIF syntax, respectively.                 *
 *                                                                    *
 * * Software claiming to read definitions from a specific data       *
 * dictionary approved by the IUCr must be able to extract any        *
 * pertinent definition which is conformant to the dictionary         *
 * definition language (DDL)[3] associated with that dictionary.      *
 *                                                                    *
 * The IUCr, through its Committee on CIF Standards, will assist      *
 * any developer to verify that software meets these conformance      *
 * conditions.                                                        *
 *                                                                    *
 * Glossary of terms                                                  *
 *                                                                    *
 * [1] CIF:  is a data file conformant to the file syntax defined     *
 * at http://www.iucr.org/iucr-top/cif/spec/index.html                *
 *                                                                    *
 * [2] STAR File:  is a data file conformant to the file syntax       *
 * defined at http://www.iucr.org/iucr-top/cif/spec/star/index.html   *
 *                                                                    *
 * [3] DDL:  is a language used in a data dictionary to define data   *
 * items in terms of "attributes". Dictionaries currently approved    *
 * by the IUCr, and the DDL versions used to construct these          *
 * dictionaries, are listed at                                        *
 * http://www.iucr.org/iucr-top/cif/spec/ddl/index.html               *
 *                                                                    *
 * Last modified: 30 September 2000                                   *
 *                                                                    *
 * IUCr Policy Copyright (C) 2000 International Union of              *
 * Crystallography                                                    *
 **********************************************************************/

#ifdef __cplusplus

extern "C" {
    
#endif
    
#include "cbf.h"
    
#include "cbf_copy.h"
#include "cbf_alloc.h"
#include "cbf_string.h"
    
#include <ctype.h>
#include <string.h>
#include <math.h>
    
    
    /* cbf_copy_cbf -- copy cbfin to cbfout */
    
    int cbf_copy_cbf(cbf_handle cbfout, cbf_handle cbfin, 
                     const int compression,
                     const int dimflag) {
        
        unsigned int blocknum, blocks;
        
        const char * datablock_name;
        
        cbf_failnez (cbf_rewind_datablock(cbfin))
        
        cbf_failnez (cbf_count_datablocks(cbfin, &blocks))
        
        for (blocknum = 0; blocknum < blocks;  blocknum++ ) {
            
            cbf_failnez (cbf_select_datablock(cbfin, blocknum))
            cbf_failnez (cbf_datablock_name(cbfin, &datablock_name))
            cbf_failnez (cbf_copy_datablock(cbfout, cbfin, datablock_name, compression, dimflag))
        }
        
        return 0;
        
    }
    
    /* cbf_copy_category -- copy the current category from cbfin
     specified category in cbfout */
    
    int cbf_copy_category(cbf_handle cbfout, cbf_handle cbfin, 
                          const char * category_name,
                          const int compression,
                          const int dimflag) {
        
        unsigned int rows, columns;
        
        unsigned int rownum, colnum;
        
        const char * column_name;
        
        const char * value;
        
        cbf_failnez(cbf_force_new_category(cbfout,category_name))
        
        cbf_failnez(cbf_count_rows(cbfin,&rows));
        
        cbf_failnez(cbf_count_columns(cbfin,&columns));
        
        /*  Transfer the column names from cbfin to cbfout */
        
        if ( ! cbf_rewind_column(cbfin) ) {
            
            do {
                
                cbf_failnez(cbf_column_name(cbfin, &column_name))
                
                cbf_failnez(cbf_new_column(cbfout, column_name))
                
            } while ( ! cbf_next_column(cbfin) );
            
            cbf_failnez(cbf_rewind_column(cbfin))
            
            cbf_failnez(cbf_rewind_row(cbfin))
        }
        
        /* Transfer to rows from cbfin to cbfout */
        
        for (rownum = 0; rownum < rows; rownum++ ) {
            
            cbf_failnez (cbf_select_row(cbfin, rownum))
            
            cbf_failnez (cbf_new_row(cbfout))
            
            cbf_rewind_column(cbfin);
            
            for (colnum = 0; colnum < columns; colnum++ ) {
                
                const char *typeofvalue;
                
                cbf_failnez (cbf_select_column(cbfin, colnum))
                
                cbf_failnez (cbf_column_name(cbfin, &column_name))
                
                if ( ! cbf_get_value(cbfin, &value) ) {
                    
                    if (compression && value && column_name && !cbf_cistrcmp("compression_type",column_name)) {
                        
                        cbf_failnez (cbf_select_column(cbfout, colnum))
                        
                        switch (compression&CBF_COMPRESSION_MASK) {
                                
                            case (CBF_NONE):
                                cbf_failnez (cbf_set_value      (cbfout,"none"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                            case (CBF_CANONICAL):
                                cbf_failnez (cbf_set_value      (cbfout,"canonical"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                            case (CBF_PACKED):
                                cbf_failnez (cbf_set_value      (cbfout,"packed"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                            case (CBF_PACKED_V2):
                                cbf_failnez (cbf_set_value      (cbfout,"packed_v2"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                            case (CBF_BYTE_OFFSET):
                                cbf_failnez (cbf_set_value      (cbfout,"byte_offsets"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                            case (CBF_PREDICTOR):
                                cbf_failnez (cbf_set_value      (cbfout,"predictor"))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                                break;
                                
                                
                            default:                                
                                cbf_failnez (cbf_set_value      (cbfout,"."))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                                break;
                        }
                        if (compression&CBF_FLAG_MASK) {
                            
                            if (compression&CBF_UNCORRELATED_SECTIONS) {
                                
                                cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                                cbf_failnez (cbf_set_value        (cbfout, "uncorrelated_sections"))
                                cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                                
                            } else if (compression&CBF_FLAT_IMAGE)  {
                                
                                cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                                cbf_failnez (cbf_set_value        (cbfout, "flat"))
                                cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                                
                            }
                        } else {
                            
                            if (!cbf_find_column(cbfout, "compression_type_flag")) {
                                cbf_failnez (cbf_set_value      (cbfout,"."))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                            }
                            
                        }
                        
                    } else  if (compression && value && column_name && !cbf_cistrcmp("compression_type_flag",column_name)) {
                        
                        if (compression&CBF_FLAG_MASK) {
                            
                            if (compression&CBF_UNCORRELATED_SECTIONS) {
                                cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                                cbf_failnez (cbf_set_value        (cbfout, "uncorrelated_sections"))
                                cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                                
                            } else if (compression&CBF_FLAT_IMAGE)  {
                                cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                                cbf_failnez (cbf_set_value        (cbfout, "flat"))
                                cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                            }
                            
                        } else {
                            
                            if (!cbf_find_column(cbfout, "compression_type_flag")) {
                                cbf_failnez (cbf_set_value      (cbfout,"."))
                                cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                            }
                            
                        }             	
                    } else {
                        
                        cbf_failnez (cbf_get_typeofvalue(cbfin, &typeofvalue))
                        cbf_failnez (cbf_select_column(cbfout, colnum))
                        cbf_failnez (cbf_set_value(cbfout, value))
                        cbf_failnez (cbf_set_typeofvalue(cbfout, typeofvalue))
                    }
                    
                } else {
                    
                    void * array;
                    
                    int binary_id, elsigned, elunsigned;
                    
                    size_t elements,elements_read, elsize;
                    
                    int minelement, maxelement;
                    
                    unsigned int cifcompression;
                    
                    int realarray;
                    
                    const char *byteorder;
                    
                    size_t dim1, dim2, dim3, padding;
                    
                    cbf_failnez(cbf_get_arrayparameters_wdims_fs(
                                                                 cbfin, &cifcompression,
                                                                 &binary_id, &elsize, &elsigned, &elunsigned,
                                                                 &elements, &minelement, &maxelement, &realarray,
                                                                 &byteorder, &dim1, &dim2, &dim3, &padding))
                    
                    if ((array=malloc(elsize*elements))) {
                        
                        cbf_failnez (cbf_select_column(cbfout,colnum))
                        
                        if (!realarray)  {
                            
                            cbf_failnez (cbf_get_integerarray(
                                                              cbfin, &binary_id, array, elsize, elsigned,
                                                              elements, &elements_read))
                            
                            if (dimflag == CBF_HDR_FINDDIMS && dim1==0) {
                                cbf_get_arraydimensions(cbfin,NULL,&dim1,&dim2,&dim3);
                            }
                            
                            cbf_failnez(cbf_set_integerarray_wdims_fs(
                                                                      cbfout, compression,
                                                                      binary_id, array, elsize, elsigned, elements,
                                                                      "little_endian", dim1, dim2, dim3, 0))
                        } else {
                            
                            cbf_failnez (cbf_get_realarray(
                                                           cbfin, &binary_id, array, elsize,
                                                           elements, &elements_read))
                            
                            if (dimflag == CBF_HDR_FINDDIMS && dim1==0) {
                                cbf_get_arraydimensions(cbfin,NULL,&dim1,&dim2,&dim3);
                            }
                            
                            cbf_failnez(cbf_set_realarray_wdims_fs(
                                                                   cbfout, compression,
                                                                   binary_id, array, elsize, elements,
                                                                   "little_endian", dim1, dim2, dim3, 0))                 	
                        }
                        
                        free(array);
                        
                    } else {
                        
                        return CBF_ALLOC;
                    }
                }
            }
        }
        
        return 0;
        
    }
    
    /* cbf_copy_datablock -- copy the current datablock from cbfin
     to the next datablock in cbfout
     */
    
    int cbf_copy_datablock (cbf_handle cbfout, cbf_handle cbfin, 
                            const char * datablock_name,
                            const int compression,
                            const int dimflag) {
        
        CBF_NODETYPE itemtype;
        
        const char *category_name;
        
        const char *saveframe_name;
        
        unsigned int itemnum, blockitems,catnum,categories;
        
        cbf_failnez (cbf_force_new_datablock(cbfout, datablock_name))
        
        if ( !cbf_rewind_blockitem(cbfin, &itemtype) ) {
            cbf_failnez (cbf_count_blockitems(cbfin, &blockitems))
            
            for (itemnum = 0; itemnum < blockitems;  itemnum++) {
                
                cbf_failnez(cbf_select_blockitem(cbfin, itemnum, &itemtype))
                
                if (itemtype == CBF_CATEGORY) {
                    
                    cbf_failnez(cbf_category_name(cbfin,&category_name))
                    cbf_failnez(cbf_copy_category(cbfout,cbfin,category_name, compression, dimflag))
                    
                } else {
                    
                    cbf_failnez(cbf_saveframe_name(cbfin,&saveframe_name))
                    cbf_force_new_saveframe(cbfout, saveframe_name);
                    
                    if ( !cbf_rewind_category(cbfin) ) {
                        
                        cbf_failnez (cbf_count_categories(cbfin, &categories))
                        
                        for (catnum = 0; catnum < categories;  catnum++) {
                            
                            cbf_select_category(cbfin, catnum);
                            cbf_category_name(cbfin,&category_name);
                            cbf_failnez(cbf_copy_category(cbfout,cbfin,category_name, compression, dimflag))
                            
                        }
                        
                    }
                    
                }
                
            }
            
        }
        
        return 0;
        
    }
    
    /* cbf_copy_value -- copy the current value from cbfin to cbfout,
       specifying the target category, column, rownum, compression, dimension details,
       element type, size and sign */
    
    int cbf_copy_value(cbf_handle cbfout, cbf_handle cbfin, 
                          const char * category_name,
                          const char * column_name,
                          const unsigned int rownum, 
                          const int compression,
                          const int dimflag,
                          const int eltype,
                          const int elsize,
                          const int elsign,
                          const double cliplow,
                          const double cliphigh) {
        
        unsigned int rows;
        
        const char * value;
        
        char * border;
        
#ifndef CBF_USE_LONG_LONG
        
        size_t lobyte, hibyte;
        
        double vallow, valhigh;
        
#endif
        
        cbf_get_local_integer_byte_order(&border);

        
        if ( ! (eltype==0 
                || eltype==CBF_CPY_SETINTEGER 
                || eltype==CBF_CPY_SETREAL)) return CBF_ARGUMENT;
        
        if ( ! (elsign==0 
                || elsign==CBF_CPY_SETUNSIGNED 
                || elsign==CBF_CPY_SETSIGNED)) return CBF_ARGUMENT;
        
        if (elsize != 0 &&
            elsize != sizeof (long int) &&
#ifdef CBF_USE_LONG_LONG
            elsize != sizeof(long long int) &&
#else
            elsize != 2* sizeof (long int) &&
#endif
            elsize != sizeof (short int) &&
            elsize != sizeof (char))
            return CBF_ARGUMENT;
        
        
        cbf_failnez(cbf_require_category(cbfout,category_name))
        
        cbf_failnez(cbf_count_rows(cbfout,&rows));
        
        while (rows < rownum+1) {
            
            cbf_failnez(cbf_new_row(cbfout))
            
            rows++;
            
        }
        
        cbf_failnez(cbf_require_column(cbfout,column_name))
        
        cbf_failnez(cbf_select_row(cbfout,rownum))
        
        if ( ! cbf_get_value(cbfin, &value) ) {
            
            if (compression && value && !cbf_cistrcmp("compression_type",column_name)) {
                
                switch (compression&CBF_COMPRESSION_MASK) {
                        
                    case (CBF_NONE):
                        cbf_failnez (cbf_set_value      (cbfout,"none"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                    case (CBF_CANONICAL):
                        cbf_failnez (cbf_set_value      (cbfout,"canonical"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                    case (CBF_PACKED):
                        cbf_failnez (cbf_set_value      (cbfout,"packed"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                    case (CBF_PACKED_V2):
                        cbf_failnez (cbf_set_value      (cbfout,"packed_v2"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                    case (CBF_BYTE_OFFSET):
                        cbf_failnez (cbf_set_value      (cbfout,"byte_offsets"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                    case (CBF_PREDICTOR):
                        cbf_failnez (cbf_set_value      (cbfout,"predictor"))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"word"))
                        break;
                        
                        
                    default:                                
                        cbf_failnez (cbf_set_value      (cbfout,"."))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                        break;
                }
                if (compression&CBF_FLAG_MASK) {
                    
                    if (compression&CBF_UNCORRELATED_SECTIONS) {
                        
                        cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                        cbf_failnez (cbf_set_value        (cbfout, "uncorrelated_sections"))
                        cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                        
                    } else if (compression&CBF_FLAT_IMAGE)  {
                        
                        cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                        cbf_failnez (cbf_set_value        (cbfout, "flat"))
                        cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                        
                    }
                } else {
                    
                    if (!cbf_find_column(cbfout, "compression_type_flag")) {
                        cbf_failnez (cbf_set_value      (cbfout,"."))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                    }
                    
                }
                
            } else  if (compression && value && !cbf_cistrcmp("compression_type_flag",column_name)) {
                
                if (compression&CBF_FLAG_MASK) {
                    
                    if (compression&CBF_UNCORRELATED_SECTIONS) {
                        cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                        cbf_failnez (cbf_set_value        (cbfout, "uncorrelated_sections"))
                        cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                        
                    } else if (compression&CBF_FLAT_IMAGE)  {
                        cbf_failnez (cbf_require_column   (cbfout, "compression_type_flag"))
                        cbf_failnez (cbf_set_value        (cbfout, "flat"))
                        cbf_failnez (cbf_set_typeofvalue  (cbfout, "word"))
                    }
                    
                } else {
                    
                    if (!cbf_find_column(cbfout, "compression_type_flag")) {
                        cbf_failnez (cbf_set_value      (cbfout,"."))
                        cbf_failnez (cbf_set_typeofvalue(cbfout,"null"))
                    }
                    
                }             	
            } else {
                
                const char *typeofvalue;
                
                cbf_failnez (cbf_get_typeofvalue(cbfin, &typeofvalue))
                cbf_failnez (cbf_set_value(cbfout, value))
                cbf_failnez (cbf_set_typeofvalue(cbfout, typeofvalue))
            }
            
        } else {
            
            void * array;
            
            int binary_id, elsigned, elunsigned;
            
            size_t elements,elements_read, oelsize;
            
            int minelement, maxelement;
            
            unsigned int cifcompression;
            
            int realarray;
            
            const char *byteorder;
            
            size_t dim1, dim2, dim3, padding;
            
            cbf_failnez(cbf_get_arrayparameters_wdims_fs(
                                                         cbfin, &cifcompression,
                                                         &binary_id, &oelsize, &elsigned, &elunsigned,
                                                         &elements, &minelement, &maxelement, &realarray,
                                                         &byteorder, &dim1, &dim2, &dim3, &padding))
            
            if (oelsize != sizeof (long int) &&
#ifdef CBF_USE_LONG_LONG
                oelsize != sizeof(long long int) &&
#else
                oelsize != 2* sizeof (long int) &&
#endif
                oelsize != sizeof (short int) &&
                oelsize != sizeof (char))
                return CBF_ARGUMENT;
            
            
            if ((array=malloc(oelsize*elements))) {
                
                size_t nelsize;
                
                int nelsigned, nelunsigned;
                
                int icount, jcount, fill;                        
                
                size_t xelsize;
                
                nelsize = oelsize;
                
                if (elsize != 0) nelsize = elsize;
                
                xelsize = nelsize;
                
                if (oelsize < nelsize) xelsize = oelsize;
                
                nelsigned = elsigned;
                
                nelunsigned = elunsigned;
                
                if (elsign & CBF_CPY_SETSIGNED) nelsigned = 1;
                
                if (elsign & CBF_CPY_SETUNSIGNED) nelunsigned = 1;
                
                if (!realarray)  {
                    
                    cbf_onfailnez (cbf_get_integerarray(
                                                        cbfin, &binary_id, array, elsize, elsigned,
                                                        elements, &elements_read), {free(array);})
                    
                    if (dimflag == CBF_HDR_FINDDIMS && dim1==0) {
                        cbf_get_arraydimensions(cbfin,NULL,&dim1,&dim2,&dim3);
                    }
                    
                    if (((eltype &(CBF_CPY_SETINTEGER)) || eltype == 0)
                        && (elsize == 0 || elsize==oelsize)
                        && (elsign == 0 || 
                            ((elsign & CBF_CPY_SETSIGNED) && elsigned) ||
                            ((elsign & CBF_CPY_SETUNSIGNED) && elunsigned))
                        && cliplow >= cliphigh) {
                        
                        cbf_onfailnez(cbf_set_integerarray_wdims_fs(
                                                                    cbfout, compression,
                                                                    binary_id, array, oelsize, elsigned, elements,
                                                                    "little_endian", dim1, dim2, dim3, 0),{free(array);} )
                        free(array);
                        
                    } else {
                        
                        void * narray;
                        
                        int loword, hiword;
                                                
                        unsigned long maxlonguint;
                        
                        double onemore;
                        
                        maxlonguint = ~0;
                        
                        onemore = ((double)maxlonguint)+1.;
                        
                        if (toupper(border[0])=='L') {
                            
                            loword = 0;
                            
                            hiword = 1;
                            
                        } else {
                            
                            loword = 1;
                            
                            hiword = 0;
                            
                        }
                        
                        if ((narray=malloc(nelsize*elements))) {
                            
                            if (cliplow < cliphigh) {
                                
                                double doval;
                                
                                for (icount = 0; icount < elements; icount++) {
                                    
                                    switch (oelsize) {
                                            
                                        case (sizeof(char)):
                                            if (elsigned) doval = (double)((signed char *)array)[icount];
                                            else doval = (double)((unsigned char *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            if (elsigned) ((signed char *)array)[icount] = (signed char)doval;
                                            else ((unsigned char *)array)[icount] = (unsigned char)doval;
                                            break;
                                            
                                        case (sizeof(short int)):
                                            if (elsigned) doval = (double)((signed short int *)array)[icount];
                                            else doval = (double)((unsigned short int *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            if (elsigned) ((signed char *)array)[icount] = (signed short int)doval;
                                            else ((unsigned char *)array)[icount] = (unsigned short int)doval;
                                            break;
                                            
                                        case (sizeof(long int)):
                                            if (elsigned) doval = (double)((signed long int *)array)[icount];
                                            else doval = (double)((unsigned long int *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            if (elsigned) ((signed char *)array)[icount] = (signed long int)doval;
                                            else ((unsigned char *)array)[icount] = (unsigned long int)doval;
                                            break;
                                            
#ifdef CBF_USE_LONG_LONG                                                        
                                        case (sizeof(long long int)):
                                            if (elsigned) doval = (double)((signed long long int *)array)[icount];
                                            else doval = (double)((unsigned long long int *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            if (elsigned) ((signed char *)array)[icount] = (signed long long int)doval;
                                            else ((unsigned char *)array)[icount] = (unsigned long long int)doval;
                                            break;
#endif
                                        default:
                                            free(narray); free(array); return CBF_ARGUMENT;
                                            
                                    }
                                    
                                }
                                
                            }
                                                        
                            if ((eltype & CBF_CPY_SETINTEGER) || eltype == 0 ) {
                                                                
                                
                                /* integer to integer conversion */
                                
                                if (toupper(border[0])=='L') {
                                    
                                        
                                        for (icount = 0; icount < elements; icount++ ) {
                                                                                        
                                            
                                            memmove(((unsigned char *)narray)+icount*elsize,((unsigned char *)array)+icount*oelsize,xelsize);


                                            if (xelsize < nelsize) {
                                                
                                                fill = 0;
                                                
                                                if (nelsigned) fill = 
                                                    (((signed char *)array)[icount*oelsize+oelsize-1]<0)?(~0):0;
                                                
                                                if (nelunsigned) 
                                                    for(jcount=0;jcount<=nelsize-oelsize;jcount++) 
                                                        ((signed char *)narray)[icount*elsize+xelsize+jcount]=fill;
                                                
                                            }
                                            
                                        } 
 
                                    } else {
                                    
                                    for (icount = 0; icount < elements; icount++ ) {
                                        
                                        for (jcount = xelsize-1; jcount>=0; jcount--) {
                                            
                                            ((unsigned char *)narray)[icount*elsize+jcount] =  ((unsigned char *)array)[icount*oelsize+jcount];
                                            
                                            if (xelsize < nelsize) {
                                                
                                                fill = 0;
                                                
                                                if (nelsigned) fill = 
                                                    (((signed char *)array)[icount*oelsize]<0)?(~0):0;
                                                
                                                if (nelunsigned) 
                                                    for(jcount=0;jcount<=nelsize-oelsize;jcount++) 
                                                        ((signed char *)narray)[icount*elsize+jcount]=fill;
                                                
                                            }
                                            
                                        }
                                        
                                    } 
                                    
                                }
                                
                                cbf_onfailnez(cbf_set_integerarray_wdims_fs(
                                                                            cbfout, compression,
                                                                            binary_id, narray, elsize, nelsigned, elements,
                                                                            "little_endian", dim1, dim2, dim3, 0), {free(array); free(narray);})
                                free(narray);
                                
                                free(array);
                                
                                
                            } else {
                                
                                /* integer to real conversion */
                                
                                double xvalue;
                                
                                switch (oelsize) {
                                        
                                    case sizeof(char):
                                        
                                        if (elsigned) {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((signed char *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        } else {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((unsigned char *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        }
                                        
                                        break;
                                        
                                    case sizeof(short int):
                                        
                                        if (elsigned) {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((signed short int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        } else {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((unsigned short int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else  { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        }
                                        
                                        break;
                                        
                                    case sizeof(long int):
                                        
                                        if (elsigned) {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((signed long int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        } else {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((unsigned long int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        }
                                        
                                        
                                        break;
#ifdef CBF_USE_LONG_LONG
                                    case sizeof(long long int):
                                        
                                        if (elsigned) {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((signed long long int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        } else {
                                            
                                            for (icount = 0; icount < elements; icount++) {
                                                
                                                xvalue = ((unsigned long long int *)array)[icount];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        }
                                        
                                        
                                        break;
                                        
#else
                                    case 2* sizeof(long int):
                                        
                                        if (elsigned) {
                                            
                                            unsigned long yvalue[2];
                                            
                                            for (icount = 0; icount < 2* elements; icount++) {
                                                
                                                yvalue[0] = ((unsigned long int *)array)[2*icount];
                                                
                                                yvalue[1] = ((unsigned long int *)array)[2*icount+1];
                                                
                                                if ((long)yvalue[hiword]>0) {
                                                    
                                                    xvalue = ((double)yvalue[hiword])*onemore+(double)yvalue[loword];
                                                    
                                                } else {
                                                    
                                                    xvalue = -((double)(-yvalue[hiword])*onemore-(double)yvalue[loword]);
                                                    
                                                }
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        } else {
                                            
                                            unsigned long yvalue[2];
                                            
                                            for (icount = 0; icount < 2* elements; icount++) {
                                                
                                                yvalue[0] = ((unsigned long int *)array)[2*icount];
                                                
                                                yvalue[1] = ((unsigned long int *)array)[2*icount+1];
                                                
                                                xvalue = ((double)yvalue[hiword])*onemore+(double)yvalue[loword];
                                                
                                                if (elsize == sizeof(double)) ((double *)narray)[icount] = xvalue;
                                                
                                                else if (elsize == sizeof(float)) ((float *)narray)[icount] = xvalue;
                                                
                                                else { free(narray); free(array); return CBF_ARGUMENT;}
                                            }
                                            
                                        }
                                        
                                        
                                        break;
                                        
                                        
#endif
                                        
                                    default:  free(narray); free(array); return CBF_ARGUMENT;
                                        
                                }
                                
                                cbf_onfailnez(cbf_set_realarray_wdims_fs(
                                                                         cbfout, compression,
                                                                         binary_id, narray, elsize, elements,
                                                                         "little_endian", dim1, dim2, dim3, 0),
                                              
                                              { free(narray); free(array);})
                                
                                free(narray);
                                
                                free(array);
                                
                            }
                            
                        } else {
                            
                            free(array);
                            
                            return CBF_ALLOC;
                        }
                    }
                    
                } else {
                    
                    cbf_onfailnez (cbf_get_realarray(
                                                     cbfin, &binary_id, array, oelsize,
                                                     elements, &elements_read), {free(array);})
                    
                    if (dimflag == CBF_HDR_FINDDIMS && dim1==0) {
                        cbf_get_arraydimensions(cbfin,NULL,&dim1,&dim2,&dim3);
                    }
                    
                    if (((eltype &(CBF_CPY_SETREAL)) || eltype == 0)
                        && (elsize == 0 || elsize==oelsize) && cliplow >= cliphigh) {
                        
                        
                        cbf_failnez(cbf_set_realarray_wdims_fs(
                                                               cbfout, compression,
                                                               binary_id, array, oelsize, elements,
                                                               "little_endian", dim1, dim2, dim3, 0)) 
                        
                        free(array);
                        
                    } else {
                        
                        void * narray;
                        
                        double valtemp;
                        
                        if ((narray=malloc(nelsize*elements))) {
                            
                            if (cliplow < cliphigh) {
                                
                                double doval;
                                
                                for (icount = 0; icount < elements; icount++) {
                                    
                                    switch (oelsize) {
                                            
                                        case (sizeof(float)):
                                            doval = (double)((float *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            ((float *)array)[icount] = (float)doval;
                                            break;
                                            
                                        case (sizeof(double)):
                                            doval = ((double *)array)[icount];
                                            if (doval < cliplow) doval = cliplow;
                                            if (doval > cliphigh) doval = cliphigh;
                                            ((double *)array)[icount] = doval;
                                            break;
                                            
                                        default:
                                            free(narray); free(array); return CBF_ARGUMENT;
                                            
                                    }
                                    
                                }
                                
                            }
                            
                            if ((eltype & CBF_CPY_SETINTEGER) || eltype == 0 ) {
                                
                                /* real to integer conversion */
                                
                                double maxval, minval;
                                
#ifndef CBF_USE_LONG_LONG
                                double onemore;
                                
                                unsigned long int maxlongval;
                                
                                maxlongval = ~0L;
                                
                                onemore = ((double)maxlongval)+1.;
#endif
                                
                                if (nelunsigned) {
                                    
                                    minval = 0.;
                                    
                                    switch( nelsize )  {
                                            
                                        case 1:  maxval = (double)(0xFF); break;
                                        case 2:  maxval = (double)(0xFFFFU); break;
                                        case 4:  maxval = (double)(0xFFFFFFFFUL); break;
                                        case 8:  maxval = ((double)(0xFFFFFFFFUL))*(2.+((double)(0xFFFFFFFFUL))); break;
                                        default: free(array); free(narray); return CBF_ARGUMENT;
                                            
                                    }
                                    
                                } else if (nelsigned) { 
                                    
                                    switch( nelsize ) {
                                            
                                        case 1:  maxval = (double)(0x7F); break;
                                        case 2:  maxval = (double)(0x7FFFU); break;
                                        case 4:  maxval = (double)(0x7FFFFFFFUL); break;
                                        case 8:  maxval = ((double)(0xFFFFFFFFUL)) +
                                            ((double)(0x7FFFFFFFL))*(1.+((double)(0xFFFFFFFFUL))); break;
                                        default: free(array); free(narray); return CBF_ARGUMENT;
                                            
                                    }
                                    
                                    minval = -maxval;
                                    
                                    if ((int)(~0)+1 == 0) minval = minval -1;
                                    
                                } else {free(array); free(narray); return CBF_ARGUMENT;}
                                
                                
                                switch( nelsize ) {
                                        
                                    case (sizeof(char)):
                                        
                                        if (oelsize == sizeof(float)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed char *)narray)[icount] = (signed char)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned char *)narray)[icount] = (unsigned char)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else if (oelsize == sizeof(double)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed char *)narray)[icount] = (signed char)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned char *)narray)[icount] = (unsigned char)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else { free(narray); free(array); return CBF_ARGUMENT;}
                                        
                                        break;
                                        
                                    case (sizeof(short int)):
                                        
                                        if (oelsize == sizeof(float)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed short int *)narray)[icount] = (signed short int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned short int *)narray)[icount] = (unsigned short int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else if (oelsize == sizeof(double)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed short int *)narray)[icount] = (signed short int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned short int *)narray)[icount] = (unsigned short int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else { free(narray); free(array); return CBF_ARGUMENT;}
                                        
                                        break;
                                        
                                    case (sizeof(long int)):
                                        
                                        if (oelsize == sizeof(float)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                                                                        
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed long int *)narray)[icount] = (signed long int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned long int *)narray)[icount] = (unsigned long int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else if (oelsize == sizeof(double)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed long int *)narray)[icount] = (signed long int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned long int *)narray)[icount] = (unsigned long int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else { free(narray); free(array); return CBF_ARGUMENT;}
                                        
                                        break;
                                        
#ifdef CBF_USE_LONG_LONG
                                    case (sizeof(long long int)):
                                        
                                        if (oelsize == sizeof(float)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed long long int *)narray)[icount] = (signed long long int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned long long int *)narray)[icount] = (unsigned long long int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else if (oelsize == sizeof(double)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((signed long long int *)narray)[icount] = (signed long long int)valtemp;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    ((unsigned long long int *)narray)[icount] = (unsigned long long int)valtemp;
                                                }
                                                
                                            }
                                            
                                        } else { free(narray); free(array); return CBF_ARGUMENT;}
                                        
                                        break;
#else
                                    case (2* sizeof(long int)):
                                        
                                        if (toupper(border[0])=='L') {
                                            
                                            lobyte = 0;
                                            
                                            hibyte = 1;
                                            
                                        } else {
                                            
                                            lobyte = 1;
                                            
                                            hibyte = 0;
                                            
                                        }
                                        
                                        
                                        if (oelsize == sizeof(float)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    vallow = fmod(valtemp,onemore);
                                                    
                                                    valhigh = (valtemp-vallow)/onemore;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+lobyte] = (unsigned long int)vallow;
                                                    
                                                    ((signed long int *)narray)[2*icount+hibyte] = (signed long int)valhigh;
                                                    
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((float *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    vallow = fmod(valtemp,onemore);
                                                    
                                                    valhigh = (valtemp-vallow)/onemore;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+lobyte] = (unsigned long int)vallow;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+hibyte] = (unsigned long int)valhigh;
                                                    
                                                }
                                                
                                            }
                                            
                                        } else if (oelsize == sizeof(double)) {
                                            
                                            if (nelsigned) {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    vallow = fmod(valtemp,onemore);
                                                    
                                                    valhigh = (valtemp-vallow)/onemore;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+lobyte] = (unsigned long int)vallow;
                                                    
                                                    ((signed long int *)narray)[2*icount+hibyte] = (signed long int)valhigh;
                                                }
                                                
                                            } else {
                                                
                                                for (icount = 0; icount < elements; icount++) {
                                                    
                                                    valtemp = ((double *)array)[icount];
                                                    
                                                    if (valtemp < minval || valtemp > maxval) {
                                                        
                                                        free(array); free(narray); return CBF_OVERFLOW;
                                                    }
                                                    
                                                    vallow = fmod(valtemp,onemore);
                                                    
                                                    valhigh = (valtemp-vallow)/onemore;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+lobyte] = (unsigned long int)vallow;
                                                    
                                                    ((unsigned long int *)narray)[2*icount+hibyte] = (signed long int)valhigh;
                                                    
                                                }
                                                
                                            }
                                            
                                        } else { free(narray); free(array); return CBF_ARGUMENT;}
                                        
                                        break;
                                        
#endif
                                        
                                    default:
                                        
                                        free(array);
                                        
                                        free(narray);
                                        
                                        return CBF_ARGUMENT;
                                        
                                }
                                

                                
                                cbf_failnez(cbf_set_integerarray_wdims_fs(
                                                                          cbfout, compression,
                                                                          binary_id, narray, nelsize, nelsigned, elements,
                                                                          "little_endian", dim1, dim2, dim3, 0))
                                free(narray);
                                
                                free(array);
                                
                                
                            } else {
                                
                                /* real to real conversion */
                                
                                switch (oelsize) {
                                        
                                    case sizeof(float):
                                        if (nelsize == sizeof(float)) {
                                            for (icount = 0; icount < elements; icount++) {
                                                ((float *)narray)[icount] = ((float *)array)[icount];
                                            }
                                        } else if (nelsize == sizeof(double)) {
                                            for (icount = 0; icount < elements; icount++) {
                                                ((double *)narray)[icount] = ((float *)array)[icount];
                                            }
                                        } else {free(array); free(narray); return CBF_ARGUMENT;}
                                        break;
                                    case sizeof(double):
                                        if (nelsize == sizeof(float)) {
                                            for (icount = 0; icount < elements; icount++) {
                                                ((float *)narray)[icount] = ((double *)array)[icount];
                                            }
                                        } else if (nelsize == sizeof(double)) {
                                            for (icount = 0; icount < elements; icount++) {
                                                ((double *)narray)[icount] = ((double *)array)[icount];
                                            }
                                        } else {free(array); free(narray); return CBF_ARGUMENT;}
                                        break;
                                    default: free(array); free(narray); return CBF_ARGUMENT;
                                }
                                
                                cbf_failnez(cbf_set_realarray_wdims_fs(
                                                                       cbfout, compression,
                                                                       binary_id, narray, nelsize, elements,
                                                                       "little_endian", dim1, dim2, dim3, 0)) 
                                
                                free(array);
                                
                                free(narray);
                                
                                return 0;
                                
                            }
                            
                        } else {
                            
                            return CBF_ALLOC;
                        }
                    }
                }
                
                
            } else {
                
                return CBF_ALLOC;
            }
        }
        
        return 0;
    }
    
    
#ifdef __cplusplus
    
}

#endif



