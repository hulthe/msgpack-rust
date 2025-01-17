//! Generic MessagePack deserialization.

use core::convert::TryInto;
use core::fmt::{self, Display, Formatter, Debug};
use core::num::TryFromIntError;
use core::str::{self, Utf8Error};

#[cfg(feature = "std")]
use std::{
    error,
    io::Cursor,
};

use serde::forward_to_deserialize_any;
use serde::de::{self, Deserialize, DeserializeSeed, Unexpected, Visitor};
#[cfg(feature = "std")]
use serde::de::DeserializeOwned;

use rmp;
use rmp::decode::{self, RmpRead, DecodeStringError, MarkerReadError, NumValueReadError, ValueReadError, RmpReadErr};
use rmp::Marker;

use crate::config::{BinaryConfig, DefaultConfig, HumanReadableConfig, SerializerConfig};
use crate::MSGPACK_EXT_STRUCT_NAME;

/// Enum representing errors that can occur while decoding MessagePack data.
#[derive(Debug)]
pub enum Error<R> {
    /// Failed to read a MessagePack value.
    InvalidValueRead(ValueReadError<R>),
    /// A mismatch occurred between the decoded and expected value types.
    TypeMismatch(Marker),
    /// A numeric cast failed due to an out-of-range error.
    OutOfRange,
    /// A decoded array did not have the enclosed expected length.
    LengthMismatch(u32),
    /// An otherwise uncategorized error occurred. See the enclosed string for
    /// details.
    Uncategorized(&'static str),
    /// A general error occurred while deserializing the expected type. See the
    /// enclosed string for details.
    #[cfg(feature = "std")]
    Syntax(String),
    #[cfg(not(feature = "std"))]
    Syntax(),
    /// An encoded string could not be parsed as UTF-8.
    Utf8Error(Utf8Error),
    /// The depth limit was exceeded.
    DepthLimitExceeded,
}

macro_rules! depth_count(
    ( $counter:expr, $expr:expr ) => {
        {
            $counter -= 1;
            if $counter == 0 {
                return Err(Error::DepthLimitExceeded)
            }
            let res = $expr;
            $counter += 1;
            res
        }
    }
);

#[cfg(feature = "std")]
impl<R: RmpReadErr> error::Error for Error<R> {
    #[cold]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::TypeMismatch(..) => None,
            Error::InvalidValueRead(..) => None,
            Error::LengthMismatch(..) => None,
            Error::OutOfRange => None,
            Error::Uncategorized(..) => None,
            Error::Syntax(..) => None,
            Error::Utf8Error(ref err) => Some(err),
            Error::DepthLimitExceeded => None,
        }
    }
}

impl<R: RmpReadErr> de::Error for Error<R> {
    #[cold]
    fn custom<T: Display>(_msg: T) -> Self {
        #[cfg(feature = "std")]
        return Error::Syntax(_msg.to_string());

        #[cfg(not(feature = "std"))]
        return Error::Syntax();
    }
}

impl<R: RmpReadErr> Display for Error<R> {
    #[cold]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Error::InvalidValueRead(ref err) => write!(fmt, "{err}"),
            Error::TypeMismatch(ref actual_marker) => {
                write!(fmt, "wrong msgpack marker {:?}", actual_marker)
            }
            Error::OutOfRange => fmt.write_str("numeric cast found out of range"),
            Error::LengthMismatch(expected_length) => write!(
                fmt,
                "array had incorrect length, expected {}",
                expected_length
            ),
            Error::Uncategorized(ref msg) => write!(fmt, "uncategorized error: {}", msg),
            #[cfg(not(feature = "std"))]
            Error::Syntax() => fmt.write_str("syntax error"),
            #[cfg(feature = "std")]
            Error::Syntax(ref msg) => fmt.write_str(msg),
            Error::Utf8Error(ref err) => write!(fmt, "string found to be invalid utf8: {}", err),
            Error::DepthLimitExceeded => fmt.write_str("depth limit exceeded"),
        }
    }
}

impl<R> From<ValueReadError<R>> for Error<R> {
    #[cold]
    fn from(err: ValueReadError<R>) -> Self {
        Self::InvalidValueRead(err)
    }
}

impl<R: RmpReadErr> From<MarkerReadError<R>> for Error<R> {
    #[cold]
    fn from(err: MarkerReadError<R>) -> Self {
        match err {
            MarkerReadError(err) => Error::InvalidValueRead(ValueReadError::InvalidMarkerRead(err)),
        }
    }
}

impl<R> From<Utf8Error> for Error<R>{
    #[cold]
    fn from(err: Utf8Error) -> Self {
        Error::Utf8Error(err)
    }
}

impl<R> From<NumValueReadError<R>> for Error<R> {
    #[cold]
    fn from(err: NumValueReadError<R>) -> Self {
        match err {
            NumValueReadError::TypeMismatch(marker) => Error::TypeMismatch(marker),
            NumValueReadError::InvalidMarkerRead(err) => Error::InvalidValueRead(ValueReadError::InvalidMarkerRead(err)),
            NumValueReadError::InvalidDataRead(err) => Error::InvalidValueRead(ValueReadError::InvalidDataRead(err)),
            NumValueReadError::OutOfRange => Error::OutOfRange,
        }
    }
}

impl<'a, R> From<DecodeStringError<'a, R>> for Error<R> {
    #[cold]
    fn from(err: DecodeStringError<'a, R>) -> Self {
        match err {
            DecodeStringError::InvalidMarkerRead(err) => Error::InvalidValueRead(ValueReadError::InvalidMarkerRead(err)),
            DecodeStringError::InvalidDataRead(err) => Error::InvalidValueRead(ValueReadError::InvalidDataRead(err)),
            DecodeStringError::TypeMismatch(marker) => Error::TypeMismatch(marker),
            DecodeStringError::BufferSizeTooSmall(..) => Error::Uncategorized("BufferSizeTooSmall"),
            DecodeStringError::InvalidUtf8(..) => Error::Uncategorized("InvalidUtf8"),
        }
    }
}

impl<R> From<TryFromIntError> for Error<R> {
    #[cold]
    fn from(_: TryFromIntError) -> Self {
        Error::OutOfRange
    }
}

/// A Deserializer that reads bytes from a buffer.
///
/// # Note
///
/// All instances of `ErrorKind::Interrupted` are handled by this function and the underlying
/// operation is retried.
#[derive(Debug)]
pub struct Deserializer<R, C = DefaultConfig> {
    rd: R,
    config: C,
    marker: Option<Marker>,
    depth: usize,
}

impl<R: RmpRead, C> Deserializer<R, C> {
    #[inline]
    fn take_or_read_marker(&mut self) -> Result<Marker, MarkerReadError<R::Error>> {
        self.marker
            .take()
            .map_or_else(|| rmp::decode::read_marker(&mut self.rd), Ok)
    }

    #[inline]
    fn peek_or_read_marker(&mut self) -> Result<Marker, MarkerReadError<R::Error>> {
        if let Some(m) = self.marker {
            Ok(m)
        } else {
            let m = rmp::decode::read_marker(&mut self.rd)?;
            Ok(*self.marker.insert(m))
        }
    }
}

#[cfg(feature = "std")]
impl<R: RmpRead> Deserializer<ReadReader<R>, DefaultConfig> {
    /// Constructs a new `Deserializer` by consuming the given reader.
    #[inline]
    pub fn new(rd: R) -> Self {
        Self {
            rd: ReadReader::new(rd),
            config: DefaultConfig,
            // Cached marker in case of deserializing optional values.
            marker: None,
            depth: 1024,
        }
    }
}

#[cfg(feature = "std")]
impl<R: RmpRead, C> Deserializer<ReadReader<R>, C> {
    /// Gets a reference to the underlying reader in this decoder.
    #[inline(always)]
    pub fn get_ref(&self) -> &R {
        &self.rd.rd
    }

    /// Gets a mutable reference to the underlying reader in this decoder.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.rd.rd
    }

    /// Consumes this deserializer returning the underlying reader.
    #[inline]
    pub fn into_inner(self) -> R {
        self.rd.rd
    }
}

impl<R: RmpRead, C: SerializerConfig> Deserializer<R, C> {
    /// Consumes this deserializer and returns a new one, which will deserialize types with
    /// human-readable representations (`Deserializer::is_human_readable` will return `true`).
    ///
    /// This is primarily useful if you need to interoperate with serializations produced by older
    /// versions of `rmp-serde`.
    #[inline]
    pub fn with_human_readable(self) -> Deserializer<R, HumanReadableConfig<C>> {
        let Deserializer { rd, config, marker, depth } = self;
        Deserializer {
            rd,
            config: HumanReadableConfig::new(config),
            marker,
            depth,
        }
    }

    /// Consumes this deserializer and returns a new one, which will deserialize types with
    /// binary representations (`Deserializer::is_human_readable` will return `false`).
    ///
    /// This is the default MessagePack deserialization mechanism, consuming the most compact
    /// representation.
    #[inline]
    pub fn with_binary(self) -> Deserializer<R, BinaryConfig<C>> {
        let Deserializer { rd, config, marker, depth } = self;
        Deserializer {
            rd,
            config: BinaryConfig::new(config),
            marker,
            depth,
        }
    }
}

#[cfg(feature = "std")]
impl<R: AsRef<[u8]>> Deserializer<ReadReader<Cursor<R>>> {
    /// Returns the current position of this deserializer, i.e. how many bytes were read.
    #[inline(always)]
    pub fn position(&self) -> u64 {
        self.rd.rd.position()
    }
}

impl<'de> Deserializer<ReadRefReader<'de>> {
    /// Constructs a new `Deserializer` from the given byte slice.
    #[inline(always)]
    pub fn from_bytes(rd: &'de [u8]) -> Self {
        Deserializer {
            rd: ReadRefReader::new(rd),
            config: DefaultConfig,
            marker: None,
            depth: 1024,
        }
    }
}

impl<'de, R: ReadSlice<'de>, C: SerializerConfig> Deserializer<R, C> {
    /// Changes the maximum nesting depth that is allowed
    #[inline(always)]
    pub fn set_max_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    fn read_str_data<V>(&mut self, len: u32, visitor: V) -> Result<V::Value, Error<R::Error>>
        where V: Visitor<'de>
    {
        match read_bin_data(&mut self.rd, len as u32)? {
            Reference::Borrowed(buf) => {
                match str::from_utf8(buf) {
                    Ok(s) => visitor.visit_borrowed_str(s),
                    Err(err) => {
                        // Allow to unpack invalid UTF-8 bytes into a byte array.
                        match visitor.visit_borrowed_bytes::<Error<R::Error>>(buf) {
                            Ok(buf) => Ok(buf),
                            Err(..) => Err(Error::Utf8Error(err)),
                        }
                    }
                }
            }
            Reference::Copied(buf) => {
                match str::from_utf8(buf) {
                    Ok(s) => visitor.visit_str(s),
                    Err(err) => {
                        // Allow to unpack invalid UTF-8 bytes into a byte array.
                        match visitor.visit_bytes::<Error<R::Error>>(buf) {
                            Ok(buf) => Ok(buf),
                            Err(..) => Err(Error::Utf8Error(err)),
                        }
                    }
                }
            }
        }
    }

    fn read_128(&mut self) -> Result<[u8; 16], Error<R::Error>> {
        let marker = self.take_or_read_marker()?;

        if marker != Marker::Bin8 {
            return Err(Error::TypeMismatch(marker));
        }

        let len = read_u8(&mut self.rd)?;

        if len != 16 {
            return Err(Error::LengthMismatch(16));
        }

        let buf = match read_bin_data(&mut self.rd, len as u32)? {
            Reference::Borrowed(buf) => buf,
            Reference::Copied(buf) => buf,
        };

        Ok(buf.try_into().unwrap())
    }
}

fn read_bin_data<'a, 'de, R: ReadSlice<'de>>(rd: &'a mut R, len: u32) -> Result<Reference<'de,'a, [u8]>, Error<R::Error>> {
    Ok(rd.read_slice(len as usize).map_err(ValueReadError::InvalidDataRead)?)
}

fn read_u8<R: RmpRead>(rd: &mut R) -> Result<u8, Error<R::Error>> {
    Ok(rd.read_u8()
        .map_err(ValueReadError::InvalidDataRead)?)
}

fn read_u16<R: RmpRead>(rd: &mut R) -> Result<u16, Error<R::Error>> {
    let mut bytes = [0u8; 2];
    rd.read_exact_buf(&mut bytes)
        .map_err(ValueReadError::InvalidDataRead)?;
    Ok(u16::from_be_bytes(bytes))
}

fn read_u32<R: RmpRead>(rd: &mut R) -> Result<u32, Error<R::Error>> {
    let mut bytes = [0u8; 4];
    rd.read_exact_buf(&mut bytes)
        .map_err(ValueReadError::InvalidDataRead)?;
    Ok(u32::from_be_bytes(bytes))
}

fn ext_len<R: RmpRead>(rd: &mut R, marker: Marker) -> Result<u32, Error<R::Error>> {
    Ok(match marker {
        Marker::FixExt1 => 1,
        Marker::FixExt2 => 2,
        Marker::FixExt4 => 4,
        Marker::FixExt8 => 8,
        Marker::FixExt16 => 16,
        Marker::Ext8 => read_u8(rd)? as u32,
        Marker::Ext16 => read_u16(rd)? as u32,
        Marker::Ext32 => read_u32(rd)? as u32,
        _ => return Err(Error::TypeMismatch(marker)),
    })
}

#[derive(Debug)]
enum ExtDeserializerState {
    New,
    ReadTag,
    ReadBinary,
}

#[derive(Debug)]
struct ExtDeserializer<'a, R, C> {
    rd: &'a mut R,
    _config: C,
    len: u32,
    state: ExtDeserializerState,
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> ExtDeserializer<'a, R, C> {
    fn new(d: &'a mut Deserializer<R, C>, len: u32) -> Self {
        ExtDeserializer {
            rd: &mut d.rd,
            _config: d.config,
            len,
            state: ExtDeserializerState::New,
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::Deserializer<'de> for ExtDeserializer<'a, R, C> {
    type Error = Error<R::Error>;

    #[inline(always)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        visitor.visit_seq(self)
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        struct identifier tuple enum ignored_any tuple_struct
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::SeqAccess<'de> for ExtDeserializer<'a, R, C> {
    type Error = Error<R::Error>;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.state {
            ExtDeserializerState::New | ExtDeserializerState::ReadTag => Ok(Some(seed.deserialize(self)?)),
            ExtDeserializerState::ReadBinary => Ok(None)
        }
    }
}


/// Deserializer for Ext SeqAccess
impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::Deserializer<'de> for &mut ExtDeserializer<'a, R, C> {
    type Error = Error<R::Error>;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        match self.state {
            ExtDeserializerState::New => {
                let tag = self.rd.read_data_i8()?;
                self.state = ExtDeserializerState::ReadTag;
                visitor.visit_i8(tag)
            }
            ExtDeserializerState::ReadTag => {
                let data = self.rd.read_slice(self.len as usize).map_err(ValueReadError::InvalidDataRead)?;
                self.state = ExtDeserializerState::ReadBinary;
                match data {
                    Reference::Borrowed(bytes) => visitor.visit_borrowed_bytes(bytes),
                    Reference::Copied(bytes) => visitor.visit_bytes(bytes),
                }
            }
            ExtDeserializerState::ReadBinary => unreachable!(),
        }
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        tuple_struct struct identifier tuple enum ignored_any
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> serde::Deserializer<'de> for &'a mut Deserializer<R, C> {
    type Error = Error<R::Error>;

    #[inline(always)]
    fn is_human_readable(&self) -> bool {
        C::is_human_readable()
    }

    #[inline(never)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        let marker = self.take_or_read_marker()?;

        match marker {
            Marker::Null => visitor.visit_unit(),
            Marker::True |
            Marker::False => visitor.visit_bool(marker == Marker::True),
            Marker::FixPos(val) => visitor.visit_u8(val),
            Marker::FixNeg(val) => visitor.visit_i8(val),
            Marker::U8 => visitor.visit_u8(self.rd.read_data_u8()?),
            Marker::U16 => visitor.visit_u16(self.rd.read_data_u16()?),
            Marker::U32 => visitor.visit_u32(self.rd.read_data_u32()?),
            Marker::U64 => visitor.visit_u64(self.rd.read_data_u64()?),
            Marker::I8 => visitor.visit_i8(self.rd.read_data_i8()?),
            Marker::I16 => visitor.visit_i16(self.rd.read_data_i16()?),
            Marker::I32 => visitor.visit_i32(self.rd.read_data_i32()?),
            Marker::I64 => visitor.visit_i64(self.rd.read_data_i64()?),
            Marker::F32 => visitor.visit_f32(self.rd.read_data_f32()?),
            Marker::F64 => visitor.visit_f64(self.rd.read_data_f64()?),
            Marker::FixStr(_) | Marker::Str8 | Marker::Str16 | Marker::Str32 => {
                let len = match marker {
                    Marker::FixStr(len) => Ok(len.into()),
                    Marker::Str8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Str16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Str32 => read_u32(&mut self.rd).map(u32::from),
                    _ => unreachable!()
                }?;
                self.read_str_data(len, visitor)
            }
            Marker::FixArray(_) |
            Marker::Array16 |
            Marker::Array32 => {
                let len = match marker {
                    Marker::FixArray(len) => len.into(),
                    Marker::Array16 => read_u16(&mut self.rd)?.into(),
                    Marker::Array32 => read_u32(&mut self.rd)?,
                    _ => unreachable!(),
                };

                depth_count!(self.depth, {
                    let mut seq = SeqAccess::new(self, len);
                    let res = visitor.visit_seq(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            }
            Marker::FixMap(_) |
            Marker::Map16 |
            Marker::Map32 => {
                let len = match marker {
                    Marker::FixMap(len) => len.into(),
                    Marker::Map16 => read_u16(&mut self.rd)?.into(),
                    Marker::Map32 => read_u32(&mut self.rd)?,
                    _ => unreachable!()
                };

                depth_count!(self.depth, {
                    let mut seq = MapAccess::new(self, len);
                    let res = visitor.visit_map(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            }
            Marker::Bin8 | Marker::Bin16 | Marker::Bin32 => {
                let len = match marker {
                    Marker::Bin8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Bin16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Bin32 => read_u32(&mut self.rd).map(u32::from),
                    _ => unreachable!()
                }?;
                match read_bin_data(&mut self.rd, len)? {
                    Reference::Borrowed(buf) => visitor.visit_borrowed_bytes(buf),
                    Reference::Copied(buf) => visitor.visit_bytes(buf),
                }
            }
            Marker::FixExt1 |
            Marker::FixExt2 |
            Marker::FixExt4 |
            Marker::FixExt8 |
            Marker::FixExt16 |
            Marker::Ext8 |
            Marker::Ext16 |
            Marker::Ext32 => {
                let len = ext_len(&mut self.rd, marker)?;
                depth_count!(self.depth, visitor.visit_newtype_struct(ExtDeserializer::new(self, len)))
            }
            Marker::Reserved => Err(Error::TypeMismatch(Marker::Reserved)),
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // # Important
        //
        // If a nested Option `o ∈ { Option<Opion<t>>, Option<Option<Option<t>>>, ..., Option<Option<...Option<t>...> }`
        // is visited for the first time, the marker (read from the underlying Reader) will determine
        // `o`'s innermost type `t`.
        // For subsequent visits of `o` the marker will not be re-read again but kept until type `t`
        // is visited.
        //
        // # Note
        //
        // Round trips of Options where `Option<t> = None` such as `Some(None)` will fail because
        // they are just seriialized as `nil`. The serialization format has probably to be changed
        // to solve this. But as serde_json behaves the same, I think it's not worth doing this.
        let marker = self.take_or_read_marker()?;

        if marker == Marker::Null {
            visitor.visit_none()
        } else {
            // Keep the marker until `o`'s innermost type `t` is visited.
            self.marker = Some(marker);
            visitor.visit_some(self)
        }
    }

    fn deserialize_enum<V>(self, _name: &str, _variants: &[&str], visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        let marker = self.peek_or_read_marker()?;
        match rmp::decode::marker_to_len(&mut self.rd, marker) {
            Ok(len) => match len {
                // Enums are either encoded as maps with a single K/V pair
                // where the K = the variant & V = associated data
                // or as just the variant
                1 => {
                    self.marker = None;
                    visitor.visit_enum(VariantAccess::new(self))
                }
                n => Err(Error::LengthMismatch(n as u32)),
            },
            // TODO: Check this is a string
            Err(_) => visitor.visit_enum(UnitVariantAccess::new(self)),
        }
    }

    fn deserialize_newtype_struct<V>(self, name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        if name == MSGPACK_EXT_STRUCT_NAME {
            let marker = self.take_or_read_marker()?;

            let len = ext_len(&mut self.rd, marker)?;
            let ext_de = ExtDeserializer::new(self, len);
            return visitor.visit_newtype_struct(ext_de);
        }

        visitor.visit_newtype_struct(self)
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // We need to special case this so that [] is treated as a unit struct when asked for,
        // but as a sequence otherwise. This is because we serialize unit structs as [] rather
        // than as 'nil'.
        match self.take_or_read_marker()? {
            Marker::Null | Marker::FixArray(0) => visitor.visit_unit(),
            marker => {
                self.marker = Some(marker);
                self.deserialize_any(visitor)
            }
        }
    }

    #[inline]
    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let buf = self.read_128()?;
        visitor.visit_i128(i128::from_be_bytes(buf))
    }

    #[inline]
    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let buf = self.read_128()?;
        visitor.visit_u128(u128::from_be_bytes(buf))
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32
        f64 char str string bytes byte_buf unit
        seq map struct identifier tuple
        tuple_struct ignored_any
    }
}

struct SeqAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
    left: u32,
}

impl<'a, R: 'a, C> SeqAccess<'a, R, C> {
    #[inline]
    fn new(de: &'a mut Deserializer<R, C>, len: u32) -> Self {
        SeqAccess {
            de,
            left: len,
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::SeqAccess<'de> for SeqAccess<'a, R, C> {
    type Error = Error<R::Error>;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where T: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            Ok(Some(seed.deserialize(&mut *self.de)?))
        } else {
            Ok(None)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct MapAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
    left: u32,
}

impl<'a, R: 'a, C> MapAccess<'a, R, C> {
    fn new(de: &'a mut Deserializer<R, C>, len: u32) -> Self {
        MapAccess {
            de,
            left: len,
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::MapAccess<'de> for MapAccess<'a, R, C> {
    type Error = Error<R::Error>;

    #[inline]
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
        where K: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            seed.deserialize(&mut *self.de).map(Some)
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
        where V: DeserializeSeed<'de>
    {
        seed.deserialize(&mut *self.de)
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct UnitVariantAccess<'a, R: 'a, C> {
    de: &'a mut Deserializer<R, C>,
}

impl<'a, R: 'a, C> UnitVariantAccess<'a, R, C> {
    pub fn new(de: &'a mut Deserializer<R, C>) -> Self {
        UnitVariantAccess { de }
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::EnumAccess<'de>
    for UnitVariantAccess<'a, R, C>
{
    type Error = Error<R::Error>;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let variant = seed.deserialize(&mut *self.de)?;
        Ok((variant, self))
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::VariantAccess<'de>
    for UnitVariantAccess<'a, R, C>
{
    type Error = Error<R::Error>;

    fn unit_variant(self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, _seed: T) -> Result<T::Value, Self::Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"struct variant",
        ))
    }
}

struct VariantAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
}

impl<'a, R: 'a, C> VariantAccess<'a, R, C> {
    pub fn new(de: &'a mut Deserializer<R, C>) -> Self {
        VariantAccess { de }
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::EnumAccess<'de> for VariantAccess<'a, R, C> {
    type Error = Error<R::Error>;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Self::Error>
        where V: de::DeserializeSeed<'de>,
    {
        Ok((seed.deserialize(&mut *self.de)?, self))
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::VariantAccess<'de> for VariantAccess<'a, R, C> {
    type Error = Error<R::Error>;

    #[inline]
    fn unit_variant(self) -> Result<(), Self::Error> {
        decode::read_nil(&mut self.de.rd)?;
        Ok(())
    }

    #[inline]
    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
        where T: DeserializeSeed<'de>
    {
        seed.deserialize(self.de)
    }

    #[inline]
    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, len, visitor)
    }

    #[inline]
    fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, fields.len(), visitor)
    }
}

/// Unification of both borrowed and non-borrowed reference types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference<'b, 'c, T: ?Sized + 'static> {
    /// The reference is pointed at data that was borrowed.
    Borrowed(&'b T),
    /// The reference is pointed at data that was copied.
    Copied(&'c T),
}

/// Extends the `Read` trait by allowing to read slices directly by borrowing bytes.
///
/// Used to allow zero-copy reading.
pub trait ReadSlice<'de>: RmpRead {
    /// Reads the exact number of bytes from the underlying byte-array.
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, Self::Error>;
}

/// Owned reader wrapper.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct ReadReader<R: RmpRead> {
    rd: R,
    buf: Vec<u8>,
}

#[cfg(feature = "std")]
impl<R: RmpRead> ReadReader<R> {
    #[inline]
    fn new(rd: R) -> Self {
        ReadReader {
            rd,
            buf: Vec::with_capacity(128),
        }
    }
}

#[cfg(feature = "std")]
impl<'de, R: RmpRead> ReadSlice<'de> for ReadReader<R> {
    #[inline]
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, R::Error> {
        self.buf = vec![0u8; len]; // TODO: this shouldn't pre-allocate, since that might be a DoS
                                   // risk
        self.rd.read_exact_buf(&mut self.buf)?;

        Ok(Reference::Copied(&self.buf[..]))
    }
}

#[cfg(feature = "std")]
impl<R: RmpRead> RmpRead for ReadReader<R> {
    type Error = R::Error;

    fn read_exact_buf(&mut self, buf: &mut [u8]) -> Result<(), Self::Error> {
        self.rd.read_exact_buf(buf)
    }
}

/// Borrowed reader wrapper.
#[derive(Debug)]
struct ReadRefReader<'a> {
    //whole_slice: &'a [u8],
    buf: &'a [u8],
}

impl<'a> ReadRefReader<'a> {
    ///// Returns the part that hasn't been consumed yet
    //pub fn remaining_slice(&self) -> &'a [u8] {
    //    self.buf
    //}
}

impl<'a> ReadRefReader<'a,> {
    #[inline]
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            //whole_slice: bytes,
            buf: bytes,
        }
    }
}

impl<'a> RmpRead for ReadRefReader<'a> {
    type Error = BytesReadError;

    #[inline]
    fn read_exact_buf(&mut self, into: &mut [u8]) -> Result<(), Self::Error> {
        if self.buf.len() < into.len() {
            return Err(BytesReadError::InsufficientBytes { expected: into.len(), actual: self.buf.len(), position: 0 });
        }
        let (a, b) = self.buf.split_at(into.len());
        self.buf = b;
        into.copy_from_slice(a);
        Ok(())
    }
}

impl<'de> ReadSlice<'de> for ReadRefReader<'de> {
    #[inline]
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, Self::Error> {
        if self.buf.len() < len {
            return Err(BytesReadError::InsufficientBytes { expected: len, actual: self.buf.len(), position: 0 });
        }
        let (a, b) = self.buf.split_at(len);
        self.buf = b;
        Ok(Reference::Borrowed(a))
    }
}

#[test]
fn test_as_ref_reader() {
    let buf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut rd = ReadRefReader::new(&buf);

    assert_eq!(rd.read_slice(1).unwrap(), Reference::Borrowed(&[0][..]));
    assert_eq!(rd.read_slice(6).unwrap(), Reference::Borrowed(&[1, 2, 3, 4, 5, 6][..]));
    assert!(rd.read_slice(5).is_err());
    assert_eq!(rd.read_slice(4).unwrap(), Reference::Borrowed(&[7, 8, 9, 10][..]));
}

/// Deserialize an instance of type `T` from an I/O stream of MessagePack.
///
/// # Errors
///
/// This conversion can fail if the structure of the Value does not match the structure expected
/// by `T`. It can also fail if the structure is correct but `T`'s implementation of `Deserialize`
/// decides that something is wrong with the data, for example required struct fields are missing.
#[inline]
#[cfg(feature = "std")]
pub fn from_read<R, T>(rd: R) -> Result<T, Error<R::Error>>
where R: RmpRead,
      T: DeserializeOwned
{
    Deserialize::deserialize(&mut Deserializer::new(rd))
}

/// Deserialize a temporary scope-bound instance of type `T` from a slice, with zero-copy if possible.
///
/// Deserialization will be performed in zero-copy manner whenever it is possible, borrowing the
/// data from the slice itself. For example, strings and byte-arrays won't copied.
///
/// # Errors
///
/// This conversion can fail if the structure of the Value does not match the structure expected
/// by `T`. It can also fail if the structure is correct but `T`'s implementation of `Deserialize`
/// decides that something is wrong with the data, for example required struct fields are missing.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate serde_derive;
///
/// // Encoded `["Bobby", 8]`.
/// let buf = [0x92, 0xa5, 0x42, 0x6f, 0x62, 0x62, 0x79, 0x8];
///
/// #[derive(Debug, Deserialize, PartialEq)]
/// struct Dog<'a> {
///    name: &'a str,
///    age: u8,
/// }
///
/// assert_eq!(Dog { name: "Bobby", age: 8 }, rmp_serde::from_slice(&buf).unwrap());
/// ```
#[inline(always)]
#[allow(deprecated)]
pub fn from_slice<'a, T>(bytes: &'a [u8]) -> Result<T, Error<BytesReadError>>
where
    T: Deserialize<'a>
{
    let mut de = Deserializer::from_bytes(bytes);
    Deserialize::deserialize(&mut de)
}

pub use rmp::decode::bytes::BytesReadError;

/*
#[inline]
#[doc(hidden)]
#[deprecated(note = "use from_slice")]
pub fn from_read_ref<'a, R, T>(rd: &'a R) -> Result<T, Error>
where
    R: AsRef<[u8]> + ?Sized,
    T: Deserialize<'a>,
{
    let mut de = Deserializer::from_read_ref(rd);
    Deserialize::deserialize(&mut de)
}
*/
