//! This crate connects Rust MessagePack library with [`serde`][serde] providing an ability to
//! easily serialize and deserialize both Rust built-in types, the standard library and custom data
//! structures.
//!
//! ## Motivating example
//!
//! ```
//! let buf = rmp_serde::to_vec(&(42, "the Answer")).unwrap();
//!
//! assert_eq!(
//!     vec![0x92, 0x2a, 0xaa, 0x74, 0x68, 0x65, 0x20, 0x41, 0x6e, 0x73, 0x77, 0x65, 0x72],
//!     buf
//! );
//!
//! assert_eq!((42, "the Answer"), rmp_serde::from_slice(&buf).unwrap());
//! ```
//!
//! # Type-based Serialization and Deserialization
//!
//! Serde provides a mechanism for low boilerplate serialization & deserialization of values to and
//! from MessagePack via the serialization API.
//!
//! To be able to serialize a piece of data, it must implement the `serde::Serialize` trait. To be
//! able to deserialize a piece of data, it must implement the `serde::Deserialize` trait. Serde
//! provides an annotation to automatically generate the code for these
//! traits: `#[derive(Serialize, Deserialize)]`.
//!
//! # Examples
//!
//! ```
//! extern crate serde;
//! #[macro_use]
//! extern crate serde_derive;
//! extern crate rmp_serde as rmps;
//!
//! use std::collections::HashMap;
//! use serde::{Deserialize, Serialize};
//! use rmps::{Deserializer, Serializer};
//!
//! #[derive(Debug, PartialEq, Deserialize, Serialize)]
//! struct Human {
//!     age: u32,
//!     name: String,
//! }
//!
//! fn main() {
//!     let mut buf = Vec::new();
//!     let val = Human {
//!         age: 42,
//!         name: "John".into(),
//!     };
//!
//!     val.serialize(&mut Serializer::new(&mut buf)).unwrap();
//! }
//! ```
//!
//! [serde]: https://serde.rs/
#![forbid(unsafe_code)]
//#![warn(missing_debug_implementations, missing_docs)] // TODO
#![cfg_attr(not(feature = "std"), no_std)]

use core::fmt::{self, Display, Formatter};
use core::str::{self, Utf8Error};

use serde::de;
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
pub use crate::decode::{from_read, Deserializer};
pub use crate::decode::from_slice;

#[allow(deprecated)]
#[cfg(feature = "std")]
pub use crate::encode::{to_vec, to_vec_named, Serializer};
pub use crate::encode::{write, write_named};

pub mod config;
pub mod decode;
pub mod encode;

/// Name of Serde newtype struct to Represent Msgpack's Ext
/// Msgpack Ext: Ext(tag, binary)
/// Serde data model: _ExtStruct((tag, binary))
/// Example Serde impl for custom type:
///
/// ```ignore
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// #[serde(rename = "_ExtStruct")]
/// struct ExtStruct((i8, serde_bytes::ByteBuf));
///
/// test_round(ExtStruct((2, serde_bytes::ByteBuf::from(vec![5]))),
///            Value::Ext(2, vec![5]));
/// ```
pub const MSGPACK_EXT_STRUCT_NAME: &str = "_ExtStruct";

/// Helper that allows both to encode and decode strings no matter whether they contain valid or
/// invalid UTF-8.
///
/// Regardless of validity the UTF-8 content this type will always be serialized as a string.
#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum Raw<'a> {
    Borrowed {
        s: Result<&'a str, (&'a [u8], Utf8Error)>,
    },

    #[cfg(feature = "std")]
    Owned {
        s: Result<String, (Vec<u8>, Utf8Error)>,
    },
}

impl<'a> Raw<'a> {
    /// Constructs a new `Raw` from the UTF-8 string.
    #[inline]
    pub fn new_borrowed(v: &'a str) -> Self {
        Self::Borrowed { s: Ok(v) }
    }
}

#[cfg(feature = "std")]
impl Raw<'_> {
    /// Constructs a new `Raw` from the UTF-8 string.
    #[inline]
    pub fn new(v: String) -> Self {
        Self::Owned { s: Ok(v) }
    }

    /// DO NOT USE. See <https://github.com/3Hren/msgpack-rust/issues/305>
    #[deprecated(note = "This feature has been removed")]
    pub fn from_utf8(v: Vec<u8>) -> Self {
        match String::from_utf8(v) {
            Ok(v) => Raw::new(v),
            Err(err) => {
                let e = err.utf8_error();
                Self::Owned {
                    s: Err((err.into_bytes(), e)),
                }
            }
        }
    }
}

impl Raw<'_> {
    /// Returns `true` if the raw is valid UTF-8.
    #[inline]
    pub fn is_str(&self) -> bool {
        self.as_str().is_some()
    }

    /// Returns `true` if the raw contains invalid UTF-8 sequence.
    #[inline]
    pub fn is_err(&self) -> bool {
        self.as_str().is_none()
    }

    /// Returns the string reference if the raw is valid UTF-8, or else `None`.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Borrowed { s: Ok(s) } => Some(s),
            #[cfg(feature = "std")]
            Self::Owned { s: Ok(ref s) } => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the underlying `Utf8Error` if the raw contains invalid UTF-8 sequence, or
    /// else `None`.
    #[inline]
    pub fn as_err(&self) -> Option<&Utf8Error> {
        match self {
            Self::Borrowed  { s : Err((_, ref err)) } => Some(err),
            #[cfg(feature = "std")]
            Self::Owned  { s : Err((_, ref err)) } => Some(err),
            _ => None,
        }
    }

    /// Returns a byte slice of this raw's contents.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed  { s : Err(ref err) } => err.0,
            Self::Borrowed { s: Ok(s) } => s.as_bytes(),

            #[cfg(feature = "std")]
            Self::Owned  { s : Err(ref err) } => &err.0,
            #[cfg(feature = "std")]
            Self::Owned { s: Ok(ref s) } => s.as_bytes(),
        }
    }

    /// Consumes this object, yielding the string if the raw is valid UTF-8, or else `None`.
    #[cfg(feature = "std")]
    #[inline]
    pub fn into_str(self) -> Option<String> {
        match self {
            Self::Owned { s } => s.ok(),
            Self::Borrowed { s: Ok(s) } => Some(s.to_string()),
            _ => None,
        }
    }

    /// Converts a `Raw` into a byte vector.
    #[cfg(feature = "std")]
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        match self{
            Self::Borrowed  { s : Err(ref err) } => err.0.to_vec(),
            Self::Borrowed { s: Ok(s) } => s.as_bytes().to_vec(),
            Self::Owned  { s : Err(err) } => err.0,
            Self::Owned { s: Ok(s) } => s.into_bytes(),
        }
    }
}

impl Serialize for Raw<'_> {
    fn serialize<S>(&self, se: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer
    {
        if let Some(s) = self.as_str() {
            se.serialize_str(s)
        } else {
            se.serialize_bytes(self.as_bytes())
        }
    }
}

struct RawVisitor;

impl<'de> de::Visitor<'de> for RawVisitor {
    type Value = Raw<'de>;

    #[cold]
    fn expecting(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        "string or bytes".fmt(fmt)
    }

    #[cfg(feature = "std")]
    #[inline]
    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(Raw::Owned { s: Ok(v) })
    }

    #[cfg(feature = "std")]
    #[inline]
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where E: de::Error
    {
        Ok(Raw::Owned { s: Ok(v.into()) })
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
        where
            E: de::Error, {
        Ok(Raw::Borrowed { s: Ok(v) })
    }

    #[cfg(feature = "std")]
    #[inline]
    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match str::from_utf8(v) {
            Ok(s) => Ok(s.into()),
            Err(err) => Err((v.into(), err)),
        };

        Ok(Raw::Owned { s })
    }

    fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
        where
            E: de::Error, {
        
        let s = match str::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(err) => Err((v, err)),
        };

        Ok(Raw::Borrowed { s })
    }

    #[cfg(feature = "std")]
    #[inline]
    fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match String::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(err) => {
                let e = err.utf8_error();
                Err((err.into_bytes(), e))
            }
        };

        Ok(Raw::Owned { s })
    }
}

impl<'de> Deserialize<'de> for Raw<'de> {
    #[inline]
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
        where D: de::Deserializer<'de>
    {
        de.deserialize_any(RawVisitor)
    }
}

/// Helper that allows both to encode and decode strings no matter whether they contain valid or
/// invalid UTF-8.
///
/// Regardless of validity the UTF-8 content this type will always be serialized as a string.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct RawRef<'a> {
    s: Result<&'a str, (&'a [u8], Utf8Error)>,
}

impl<'a> RawRef<'a> {
    /// Constructs a new `RawRef` from the UTF-8 string.
    #[inline]
    pub fn new(v: &'a str) -> Self {
        Self { s: Ok(v) }
    }

    #[deprecated(note = "This feature has been removed")]
    pub fn from_utf8(v: &'a [u8]) -> Self {
        match str::from_utf8(v) {
            Ok(v) => RawRef::new(v),
            Err(err) => {
                Self {
                    s: Err((v, err))
                }
            }
        }
    }

    /// Returns `true` if the raw is valid UTF-8.
    #[inline]
    pub fn is_str(&self) -> bool {
        self.s.is_ok()
    }

    /// Returns `true` if the raw contains invalid UTF-8 sequence.
    #[inline]
    pub fn is_err(&self) -> bool {
        self.s.is_err()
    }

    /// Returns the string reference if the raw is valid UTF-8, or else `None`.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self.s {
            Ok(s) => Some(s),
            Err(..) => None,
        }
    }

    /// Returns the underlying `Utf8Error` if the raw contains invalid UTF-8 sequence, or
    /// else `None`.
    #[inline]
    pub fn as_err(&self) -> Option<&Utf8Error> {
        match self.s {
            Ok(..) => None,
            Err((_, ref err)) => Some(err),
        }
    }

    /// Returns a byte slice of this raw's contents.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self.s {
            Ok(s) => s.as_bytes(),
            Err((bytes, _err)) => bytes,
        }
    }
}

impl<'a> Serialize for RawRef<'a> {
    fn serialize<S>(&self, se: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.s {
            Ok(ref s) => se.serialize_str(s),
            Err((ref b, ..)) => se.serialize_bytes(b),
        }
    }
}

struct RawRefVisitor;

impl<'de> de::Visitor<'de> for RawRefVisitor {
    type Value = RawRef<'de>;

    #[cold]
    fn expecting(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        "string or bytes".fmt(fmt)
    }

    #[inline]
    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
        where E: de::Error
    {
        Ok(RawRef { s: Ok(v) })
    }

    #[inline]
    fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match str::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(err) => Err((v, err)),
        };

        Ok(RawRef { s })
    }
}

impl<'de> Deserialize<'de> for RawRef<'de> {
    #[inline]
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
        where D: de::Deserializer<'de>
    {
        de.deserialize_any(RawRefVisitor)
    }
}
