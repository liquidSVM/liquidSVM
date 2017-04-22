-- Copyright 2015-2017 Philipp Thomann
--
-- This file is part of liquidSVM.
--
-- liquidSVM is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as
-- published by the Free Software Foundation, either version 3 of the
-- License, or (at your option) any later version.
--
-- liquidSVM is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.

-- This is a custom writer for pandoc.  It produces output
-- that can be used for roxygen2 type comments
-- although the "#' " has to be inserted at the beginning of every line afterwards.
-- 
-- Invoke with: pandoc -t md2rd.lua
--
-- Note:  you need not have lua installed on your system to use this
-- custom writer.  However, if you do have lua installed, you can
-- use it to test changes to the script.  'lua sample.lua' will
-- produce informative error messages if your code contains
-- syntax errors.
--
-- We just changed the sample.lua to commodate our needs - 
-- there is also lots of HTML-production included which we only will upgrad
-- in a lazy fashion.
--

-- Character escaping
local function escape(s, in_attribute)
  return s:gsub("[%\\]",
    function(x)
      if x == '%' then
        return '\%'
      elseif x == '\\>' then
        return '\\\\'
      else
        return x
      end
    end)
end

-- Helper function to convert an attributes table into
-- a string that can be put into HTML tags.
local function attributes(attr)
  local attr_table = {}
  for x,y in pairs(attr) do
    if y and y ~= "" then
      table.insert(attr_table, ' ' .. x .. '="' .. escape(y,true) .. '"')
    end
  end
  return table.concat(attr_table)
end

-- Run cmd on a temporary file containing inp and return result.
local function pipe(cmd, inp)
  local tmp = os.tmpname()
  local tmph = io.open(tmp, "w")
  tmph:write(inp)
  tmph:close()
  local outh = io.popen(cmd .. " " .. tmp,"r")
  local result = outh:read("*all")
  outh:close()
  os.remove(tmp)
  return result
end

-- Table to store footnotes, so they can be included at the end.
local notes = {}

-- Blocksep is used to separate block elements.
function Blocksep()
  return ""
end

-- This function is called once for the whole document. Parameters:
-- body is a string, metadata is a table, variables is a table.
-- One could use some kind of templating
-- system here; this just gives you a simple standalone HTML file.
function Doc(body, metadata, variables)
  local buffer = {}
  local function add(s)
    table.insert(buffer, s)
  end
  if metadata['title'] and metadata['title'] ~= "" then
    add('<h1 class="title">' .. metadata['title'] .. '</h1>')
  end
  for _, author in pairs(metadata['author'] or {}) do
    add('<h2 class="author">' .. author .. '</h2>')
  end
  if metadata['date'] and metadata['date'] ~= "" then
    add('<h3 class="date">' .. metadata.date .. '</h3>')
  end
  --add("@details")
  --add("@section Configuration Parameters: {")
  --add("{")
  add(body)
  --add("}")
  if #notes > 0 then
    add('<ol class="footnotes">')
    for _,note in pairs(notes) do
      add(note)
    end
    add('</ol>')
  end
  return table.concat(buffer,'\n')
end

-- The functions that follow render corresponding pandoc elements.
-- s is always a string, attr is always a table of attributes, and
-- items is always an array of strings (the items in a list).
-- Comments indicate the types of other variables.

function Str(s)
  return escape(s)
end

function Space()
  return " "
end

function LineBreak()
  return "\\\\"
end

function Emph(s)
  return "\\em{" .. s .. "}"
end

function Strong(s)
  return "\\textbf{" .. s .. "}"
end

function no_Subscript(s)
  return "<sub>" .. s .. "</sub>"
end

function no_Superscript(s)
  return "<sup>" .. s .. "</sup>"
end

function no_SmallCaps(s)
  return '<span style="font-variant: small-caps;">' .. s .. '</span>'
end

function no_Strikeout(s)
  return '<del>' .. s .. '</del>'
end

function no_Link(s, src, tit)
  return "<a href='" .. escape(src,true) .. "' title='" ..
         escape(tit,true) .. "'>" .. s .. "</a>"
end

function no_Image(s, src, tit)
  return "<img src='" .. escape(src,true) .. "' title='" ..
         escape(tit,true) .. "'/>"
end

function Code(s, attr)
  return "\\code{" .. escape(s) .. "}"
end

function InlineMath(s)
  return "\\eqn{" .. escape(s) .. "}"
end

function DisplayMath(s)
  return "\\deqn{" .. escape(s) .. "}"
end

function RawInline(format, s)
  return escape(s)
end

function no_Note(s)
  local num = #notes + 1
  -- insert the back reference right before the final closing tag.
  s = string.gsub(s,
          '(.*)</', '%1 <a href="#fnref' .. num ..  '">&#8617;</a></')
  -- add a list item with the note to the note table.
  table.insert(notes, '<li id="fn' .. num .. '">' .. s .. '</li>')
  -- return the footnote reference, linked to the note.
  return '<a id="fnref' .. num .. '" href="#fn' .. num ..
            '"><sup>' .. num .. '</sup></a>'
end

function no_Span(s, attr)
  return "<span" .. attributes(attr) .. ">" .. s .. "</span>"
end

function no_Cite(s)
  return "<span class=\"cite\">" .. s .. "</span>"
end

function Plain(s)
  return s
end

function Para(s)
  return "\n" .. s .. "\n"
end

-- lev is an integer, the header level.
function Header(lev, s, attr)
  if lev == 4 then
    return "\n\\bold{Parameter \\code{" .. s .. "}:}"
  elseif lev==2 then
    return "\n@section " .. s .. ":"
  else
    return ""
  end
end

function BlockQuote(s)
  return "\\preformatted{" .. escape(s) .. "}\n"
  --return "<blockquote>\n" .. s .. "\n</blockquote>"
end

function no_HorizontalRule()
  return "<hr/>"
end

function CodeBlock(s, attr)
  return "\\preformatted{" .. escape(s) .. "}\n"
end

function BulletList(items)
  local buffer = {}
  for _, item in pairs(items) do
    table.insert(buffer, "\\item " .. item .. "\n")
  end
  return "\\itemize{\n" .. table.concat(buffer, "\n") .. "}\n"
end

function OrderedList(items)
  local buffer = {}
  for _, item in pairs(items) do
    table.insert(buffer, "\\item " .. item .. "")
  end
  return "\\enumerate{\n" .. table.concat(buffer, "\n") .. "}\n"
end

-- Revisit association list STackValue instance.
function DefinitionList(items)
  local buffer = {}
  for _,item in pairs(items) do
    for k, v in pairs(item) do
      table.insert(buffer,"\\item{" .. k .. "}{" ..
                        table.concat(v,"\n") .. "}\n")
    end
  end
  return "\n\\describe{\n" .. table.concat(buffer, "\n") .. "}\n"
end

-- Convert pandoc alignment to something HTML can use.
-- align is AlignLeft, AlignRight, AlignCenter, or AlignDefault.
function no_html_align(align)
  if align == 'AlignLeft' then
    return 'left'
  elseif align == 'AlignRight' then
    return 'right'
  elseif align == 'AlignCenter' then
    return 'center'
  else
    return 'left'
  end
end

-- Caption is a string, aligns is an array of strings,
-- widths is an array of floats, headers is an array of
-- strings, rows is an array of arrays of strings.
function Table(caption, aligns, widths, headers, rows)
  local buffer = {}
  local function add(s)
    table.insert(buffer, s)
  end
  local col_format = ""
  for i=1,table.getn(headers) do
    col_format = col_format .. "l"
  end
  add("\\tabular{" .. col_format .. "}{")
  if caption ~= "" then
    add("<caption>" .. caption .. "</caption>")
  end
  if widths and widths[1] ~= 0 then
    for _, w in pairs(widths) do
      add('<col width="' .. string.format("%d%%", w * 100) .. '" />')
    end
  end
  local header_row = {}
  local empty_header = true
  for i, h in pairs(headers) do
    --local align = html_align(aligns[i])
    if i<table.getn(headers) then
      table.insert(header_row,'' .. h .. ' \\tab ')
    else
      table.insert(header_row,'' .. h .. '')
    end
    empty_header = empty_header and h == ""
  end
  if empty_header then
    head = ""
  else
    add('')
    for _,h in pairs(header_row) do
      add(h)
    end
    add('\\cr')
  end
  local class = "even"
  for _, row in pairs(rows) do
    add('')
    for i,c in pairs(row) do
      if i < table.getn(row) then
        add('' .. c .. ' \\tab ')
      else
        add('' .. c .. '')
      end
    end
    add('\\cr\n')
  end
  add('}')
  return table.concat(buffer,'\n')
end

function no_Div(s, attr)
  return "<div" .. attributes(attr) .. ">\n" .. s .. "</div>"
end

-- The following code will produce runtime warnings when you haven't defined
-- all of the functions you need for the custom writer, so it's useful
-- to include when you're working on a writer.
local meta = {}
meta.__index =
  function(_, key)
    io.stderr:write(string.format("WARNING: Undefined function '%s'\n",key))
    return function() return "" end
  end
setmetatable(_G, meta)

